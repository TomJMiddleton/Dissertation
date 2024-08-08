from NERParser import SQLiteDataset
import sqlite3
import torch
import faiss
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

def PopulateVecEmbeddingsDB(db_path, bi_encoder, bi_tokenizer, output_index_path, M=32, efConstruction=200, batch_size=32):
    # Load data from SQLite
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Determine embedding dimensionality
    embedding_dim = bi_encoder.get_sentence_embedding_dimension()

    # Initialize HNSW index
    index = faiss.IndexHNSWFlat(embedding_dim, M)
    index.hnsw.efConstruction = efConstruction

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Add a column for the Faiss index if it doesn't already exist
    cur.execute("ALTER TABLE Sentences ADD COLUMN FaissIndex INTEGER")
    conn.commit()

    # Iterate over the data and encode in batches
    for batch in tqdm(dataloader, desc="Processing Embeddings"):
        doc_ids, sentences = batch

        # Generate embeddings using the bi-encoder
        embeddings = bi_encoder.encode(sentences, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()  # Convert to NumPy array

        # Add embeddings to Faiss index and update the database
        for i, embedding in enumerate(embeddings):
            faiss_idx = index.ntotal  # Get the current index in Faiss
            index.add(np.expand_dims(embedding, axis=0))  # Add the embedding to the Faiss index

            # Update the Faiss index in the SQLite database
            cur.execute("UPDATE Sentences SET FaissIndex = ? WHERE SentenceID = ?", (faiss_idx, doc_ids[i]))

    # Save the Faiss index to disk
    faiss.write_index(index, output_index_path)

    # Commit and close SQLite connection
    conn.commit()
    conn.close()

    print(f"Finished processing. Faiss index saved to {output_index_path}")



if __name__ == "main":
    model_name = 'dunzhang/stella_en_1.5B_v5'
    db_path = './Datasets/Database/NewsGroupDB3.db'
    vec_db_path = './Datasets/Database/NGFAISSVec.index'
    dataset = SQLiteDataset(db_path)
    bi_tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    bi_encoder = AutoModelForCausalLM.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
    embedding_dim = 384

    PopulateVecEmbeddingsDB(db_path,
                            bi_encoder,
                            bi_tokenizer,
                            vec_db_path,
                            M=32,
                            efConstruction=200,
                            batch_size=32)
