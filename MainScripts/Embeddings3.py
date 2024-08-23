import numpy as np
from annoy import AnnoyIndex
from NERParser import SQLiteDataset
from SentenceBiEncoderModel import SentenceBiEncoder
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

def PopulateVecEmbeddingsDB(db_path, output_index_path, batch_size=32):
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    bi_encoder = SentenceBiEncoder()
    
    all_embeddings = []
    index = AnnoyIndex(1024, 'angular')

    current_doc_id = 1
    curr_embeddings = []
    for batch in tqdm(dataloader, desc="Processing Embeddings"):
        doc_ids, chunks = batch
        doc_ids = doc_ids.tolist()
        last_doc_id = doc_ids[-1]

        embeddings = bi_encoder.EncodeEmbeddings(chunks)

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
    normalized_embeddings = [v / np.linalg.norm(v) for v in all_embeddings]
    print(" All embeddings have been processed")

    print("---ADDING EMBEDDINGS---")
    for i, norm_embedding in enumerate(normalized_embeddings):
        index.add_item(i, norm_embedding.tolist())

    print("---BUILDING INDEX---")
    index.build(500, n_jobs=-1)

    print("---EXPORTING INDEX AND EMBEDDINGS---")
    index.save(output_index_path)

    print(f"Finished processing. Annoy index saved to {output_index_path}")



if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB3.db'
    vec_db_path = './Datasets/FinalDB/NGAnnoyVec.ann'
    
    print(" --------------------- \n S-Transformer instantiated \n Begin encoding embeddings \n --------------------- \n")
    PopulateVecEmbeddingsDB(db_path,
                            vec_db_path,
                            batch_size=16)