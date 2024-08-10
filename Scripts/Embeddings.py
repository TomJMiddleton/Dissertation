from NERParser import SQLiteDataset, SQLiteDatabase
import sqlite3
from contextlib import closing
from collections import defaultdict
import faiss
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

def CheckFAISSIndexColExists(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("PRAGMA table_info(Documents)")
            if "FaissIndex" not in [column[1] for column in cur.fetchall()]:
                cur.execute("ALTER TABLE Documents ADD COLUMN FaissIndex INTEGER")

def PopulateVecEmbeddingsDB(db_path, bi_encoder, output_index_path, embed_dim = 1024, M=32, efConstruction=40, batch_size=32):
    CheckFAISSIndexColExists(db_path)
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_entry = SQLiteDatabase(db_path)

    # Initialize HNSW index
    index = faiss.IndexHNSWFlat(embed_dim, M)
    index.hnsw.efConstruction = efConstruction
    index.verbose = True

    
    current_doc_id = 1
    curr_embeddings = []
    for batch in tqdm(dataloader, desc="Processing Embeddings"):
        doc_ids, chunks = batch
        doc_ids = doc_ids.tolist()
        last_doc_id = doc_ids[-1]

        
        embeddings = bi_encoder.encode(chunks)

        # Group embeddings by DocID
        doc_embeddings = defaultdict(list)
        doc_embeddings[current_doc_id] = curr_embeddings
        for doc_id, embedding in zip(doc_ids, embeddings):
            doc_embeddings[doc_id].append(embedding)

        batch_updates = []
        print("Concatenated DocIDs")
        for doc_id, emb_list in doc_embeddings.items():
            # Carry final embeddings to next batch
            if doc_id == last_doc_id:
                curr_embeddings = doc_embeddings[doc_id]
                continue
            aggregated_embedding = np.mean(emb_list, axis=0)
            aggregated_embedding = aggregated_embedding.reshape(1, -1)
            if aggregated_embedding.shape[1] != embed_dim:  
                raise ValueError(f"Embedding dimension mismatch. Expected {embed_dim}, got {aggregated_embedding.shape[1]}")

            try:
                index.add(aggregated_embedding)
            except Exception as e:
                print(f"An error occurred: {e}")
            batch_updates.append((doc_id-1, doc_id))
        
        print("Batch X done")
        current_doc_id = last_doc_id
        data_entry.AddFaissIdxToDB(batch_updates)

    # Catch any overflow
    aggregated_embedding = np.mean(curr_embeddings, axis=0)
    index.add(np.expand_dims(aggregated_embedding, axis=0))
    faiss_idx = index.ntotal - 1
    data_entry.AddFaissIdxToDB([(faiss_idx, current_doc_id)])
    print("Overflow Added")

    # Export vec store
    faiss.write_index(index, output_index_path)
    print(f"Finished processing. Faiss index saved to {output_index_path}")



if __name__ == "__main__":
    model_name = 'dunzhang/stella_en_1.5B_v5'
    db_path = './Datasets/Database/NewsGroupDB3.db'
    vec_db_path = './Datasets/Database/NGFAISSVec.index'
    bi_encoder = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    print(" --------------------- \n S-Transformer instantiated \n Begin encoding embeddings \n --------------------- \n")
    PopulateVecEmbeddingsDB(db_path,
                            bi_encoder,
                            vec_db_path,
                            M=32,
                            efConstruction=200,
                            batch_size=4)
