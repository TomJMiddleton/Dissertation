from NERParser import SQLiteDataset
from SentenceBiEncoderModel import SentenceBiEncoder
from collections import defaultdict
import faiss
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

def PopulateVecEmbeddingsDB(db_path, output_index_path, batch_size=32):
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    bi_encoder = SentenceBiEncoder()
    # Create FAISS index with Inner Product metric
    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    # ---PARAMETERS---
    d = 1024  # Dimension of embeddings
    M = 64    # Number of sub-quantizers
    D = 4 * M # Dimension after OPQ 
    K = 550   # Number of clusters for IVF 

    # Create the index using the index factory
    index_string = f"L2norm,OPQ{M}_{D},IVF{K},PQ{M}"
    index = faiss.index_factory(d, index_string, faiss.METRIC_INNER_PRODUCT)
    
    all_embeddings = []

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

        # Aggregate embeddings and append to list
        for doc_id, emb_list in doc_embeddings.items():
            # Carry final embeddings to next batch
            if doc_id == last_doc_id:
                curr_embeddings = doc_embeddings[doc_id]
                continue
            aggregated_embedding = np.mean(emb_list, axis=0)
            aggregated_embedding = aggregated_embedding
            all_embeddings.append(aggregated_embedding)
        current_doc_id = last_doc_id

    # Catch any overflow
    aggregated_embedding = np.mean(emb_list, axis=0)
    aggregated_embedding = aggregated_embedding
    all_embeddings.append(aggregated_embedding)

    all_embeddings = np.asarray(all_embeddings)
    print(" All embeddings have been processed")

    print("---TRAINING INDEX---")
    index.train(all_embeddings)

    print("---ADDING EMBEDDINGS---")
    index.add(all_embeddings)

    print("---EXPORTING INDEX AND EMBEDDINGS---")
    faiss.write_index(index, output_index_path)
    print(f"Finished processing. Faiss index saved to {output_index_path}")



if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB3.db'
    vec_db_path = './Datasets/Database/NGFAISSVec.faiss'
    
    print(" --------------------- \n S-Transformer instantiated \n Begin encoding embeddings \n --------------------- \n")
    PopulateVecEmbeddingsDB(db_path,
                            vec_db_path,
                            batch_size=16)
