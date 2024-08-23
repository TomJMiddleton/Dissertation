import torch
from torch.distributions import normal, uniform
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
from SentenceBiEncoderModel import SentenceBiEncoder
from NERParser import SQLiteDataset
from RVQModel import RVQ


def train_rvq(embeddings, num_stages, vq_bitrate_per_stage, num_epochs, batch_size, learning_rate, replacement_num_batches = 1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = embeddings.shape[1]
    
    vector_quantizer = RVQ(num_stages, vq_bitrate_per_stage, embedding_dim, device=device)
    vector_quantizer.to(device)

    optimizer = optim.Adam(vector_quantizer.parameters(), lr=learning_rate)
    num_training_batches = num_epochs * (len(embeddings) // batch_size)
    print_interval = 500
    vq_loss_accumulator = 0

    for iter in tqdm(range(num_training_batches), desc="Training RVQ"):
        batch_start = (iter * batch_size) % len(embeddings)
        batch_end = min(batch_start + batch_size, len(embeddings))
        data_batch = torch.tensor(embeddings[batch_start:batch_end], dtype=torch.float32).to(device)

        optimizer.zero_grad()

        quantized_batch, used_codeword_idx = vector_quantizer(data_batch, train_mode=True)
        vq_loss = F.mse_loss(data_batch, quantized_batch)

        vq_loss.backward()
        optimizer.step()

        # save and print logs
        if (iter+1) % print_interval == 0:
            vq_loss_average = vq_loss_accumulator / print_interval
            vq_loss_accumulator = 0
            print("training iter:{}, vq loss:{:.6f}".format(iter + 1, vq_loss_average))

        # codebook replacement
        if ((iter + 1) % replacement_num_batches == 0) & (iter <= num_training_batches - 2*replacement_num_batches):
            vector_quantizer.replace_unused_codebooks(replacement_num_batches)
    
    return vector_quantizer

def PopulateVecEmbeddingsDB(db_path, export_path, rvq_params, batch_size=32):
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    bi_encoder = SentenceBiEncoder()
    
    all_embeddings = []

    print("---GENERATING EMBEDDINGS---")
    for batch in tqdm(dataloader, desc="Processing Embeddings"):
        _, chunks = batch
        embeddings = bi_encoder.EncodeEmbeddings(chunks)
        
        all_embeddings.extend(embeddings)

    normalized_embeddings = np.array([v / np.linalg.norm(v) for v in all_embeddings])
    print("All embeddings have been processed")

    print("---TRAINING RVQ MODEL---")
    rvq_model = train_rvq(normalized_embeddings, **rvq_params)

    print("---COMPRESSING EMBEDDINGS---")
    compressed_embeddings = rvq_model.compress_embeddings(normalized_embeddings)

    print("---CREATING ANNOY INDEX---")
    compressed_dim = rvq_params['num_stages']
    index = AnnoyIndex(compressed_dim, 'hamming')

    for i, compressed_emb in enumerate(compressed_embeddings):
        index.add_item(i, compressed_emb.flatten())

    print("---BUILDING INDEX---")
    index.build(750, n_jobs=-1)

    print("---EXPORTING INDEX AND MODELS---")
    index.save(export_path + 'NGRVQVec.ann')
    torch.save(rvq_model.state_dict(), export_path + 'RVQ_Model.pt')
    np.save(export_path + '.original_embeddings.npy', normalized_embeddings)

    print(f"Finished processing. Annoy index saved to {export_path}NGRVQVec.ann")
    print(f"RVQ model saved to {export_path}RVQ_Model.pt")
    print(f"Original embeddings saved to {export_path}.original_embeddings.npy")

if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB3.db'
    export_path = './Datasets/FinalDB/'
    
    rvq_params = {
        "num_stages": 4,
        "vq_bitrate_per_stage": 6,
        "num_epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-3
    }

    PopulateVecEmbeddingsDB(db_path, export_path, rvq_params)

# Function to perform similarity search
def search_similar(query_embedding, rvq_model, annoy_index, n_results=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        compressed_query = rvq_model.encode(query_tensor).cpu().numpy().flatten()
    
    similar_indices = annoy_index.get_nns_by_vector(compressed_query, n_results)
    return similar_indices


# rvq_model = RVQ(num_stages, vq_bitrate_per_stage, embedding_dim)
# rvq_model.load_state_dict(torch.load('path/to/rvq_model.pth'))
# annoy_index = AnnoyIndex(compressed_dim, 'angular')
# annoy_index.load('path/to/compressed_embeddings.ann')
# similar_indices = search_similar(query_embedding, rvq_model, annoy_index)