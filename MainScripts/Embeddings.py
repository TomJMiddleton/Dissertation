from NERParser import SQLiteDataset
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
            cur.execute("ALTER TABLE Documents ADD COLUMN FaissIndex INTEGER")

def PopulateVecEmbeddingsDB(db_path, bi_encoder, output_index_path, embed_dim = 1024, M=32, efConstruction=200, batch_size=32):
    # Load data from SQLite
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize HNSW index
    index = faiss.IndexHNSWFlat(embed_dim, M)
    index.hnsw.efConstruction = efConstruction

    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            for batch in tqdm(dataloader, desc="Processing Embeddings"):
                doc_ids, chunks = batch

                # Generate embeddings
                embeddings = bi_encoder.encode(chunks, convert_to_tensor=True)
                embeddings = embeddings.cpu().numpy()

                # Group embeddings by DocID
                doc_embeddings = defaultdict(list)
                for doc_id, embedding in zip(doc_ids, embeddings):
                    doc_embeddings[doc_id].append(embedding)

                batch_updates = []
                for doc_id, emb_list in doc_embeddings.items():
                    aggregated_embedding = np.mean(emb_list, axis=0)
                    index.add(np.expand_dims(aggregated_embedding, axis=0))
                    faiss_idx = index.ntotal - 1
                    batch_updates.append((faiss_idx, doc_id))

                # Update DB with index for Doc:embedding
                cur.executemany("UPDATE Documents SET FaissIndex = ? WHERE DocID = ?", batch_updates)

    # Export vec store
    faiss.write_index(index, output_index_path)
    print(f"Finished processing. Faiss index saved to {output_index_path}")



if __name__ == "main":
    model_name = 'dunzhang/stella_en_1.5B_v5'
    db_path = './Datasets/Database/NewsGroupDB3.db'
    vec_db_path = './Datasets/Database/NGFAISSVec.index'
    dataset = SQLiteDataset(db_path)
    bi_encoder = SentenceTransformer(model_name, trust_remote_code=True).cuda()

    PopulateVecEmbeddingsDB(db_path,
                            bi_encoder,
                            vec_db_path,
                            M=32,
                            efConstruction=200,
                            batch_size=32)
