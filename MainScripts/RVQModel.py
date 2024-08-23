import torch
from torch.distributions import normal, uniform
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class RVQ(torch.nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, data_dim, discard_threshold=0.002, device=torch.device('cpu')):
        super(RVQ, self).__init__()

        self.num_stages = num_stages
        self.num_codebooks = int(2 ** vq_bitrate_per_stage)
        self.data_dim = data_dim
        self.eps = 1e-12
        self.device = device
        self.dtype = torch.float32
        self.normal_dist = normal.Normal(0, 1)
        self.discard_threshold = discard_threshold

        initial_codebooks = torch.zeros(self.num_stages, self.num_codebooks, self.data_dim, device=self.device)

        for k in range(num_stages):
            initial_codebooks[k] = uniform.Uniform(-1 / self.num_codebooks, 1 / self.num_codebooks).sample(
                [self.num_codebooks, self.data_dim])

        self.codebooks = torch.nn.Parameter(initial_codebooks, requires_grad=True)
        self.codebooks_used = torch.zeros((num_stages, self.num_codebooks), dtype=torch.int32, device=self.device)

    def forward(self, input_data, train_mode):
        quantized_input_list = []
        remainder_list = []
        min_indices_list = []

        remainder_list.append(input_data)

        for i in range(self.num_stages):
            quantized_input, remainder, min_indices = self.hard_vq(remainder_list[i], self.codebooks[i])

            quantized_input_list.append(quantized_input)
            remainder_list.append(remainder)
            min_indices_list.append(min_indices)

        final_input_quantized = sum(quantized_input_list[:])

        final_input_quantized_nsvq = self.noise_substitution_vq(input_data, final_input_quantized)

        with torch.no_grad():
            for i in range(self.num_stages):
                self.codebooks_used[i, min_indices_list[i]] += 1

        if train_mode:
            return final_input_quantized_nsvq, self.codebooks_used.cpu().numpy()
        else:
            return final_input_quantized.detach(), self.codebooks_used.cpu().numpy()

    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()
        vq_error = ((norm_hard_quantized_input / norm_random_vector + self.eps) * random_vector)
        quantized_input = input_data + vq_error
        return quantized_input

    def hard_vq(self, input_data, codebooks):
        distances = (torch.sum(input_data ** 2, dim=1, keepdim=True)
                     - 2 * (torch.matmul(input_data, codebooks.t()))
                     + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True))
        min_indices = torch.argmin(distances, dim=1)
        quantized_input = codebooks[min_indices]
        remainder = input_data - quantized_input
        return quantized_input, remainder, min_indices

    def replace_unused_codebooks(self, num_batches):
        with torch.no_grad():
            for k in range(self.num_stages):
                unused_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) < self.discard_threshold)[0]
                used_indices = torch.where((self.codebooks_used[k].cpu() / num_batches) >= self.discard_threshold)[0]

                unused_count = unused_indices.shape[0]
                used_count = used_indices.shape[0]

                if used_count == 0:
                    print(f'####### used_indices equals zero / shuffling whole codebooks ######')
                    self.codebooks[k] += self.eps * torch.randn(self.codebooks[k].size(), device=self.device).clone()
                else:
                    used = self.codebooks[k, used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(int((unused_count / (used_count + self.eps)) + 1), 1)
                        used_codebooks = used_codebooks[torch.randperm(used_codebooks.shape[0])]
                    else:
                        used_codebooks = used[torch.randperm(used.shape[0])]

                    self.codebooks[k, unused_indices] *= 0
                    self.codebooks[k, unused_indices] += used_codebooks[range(unused_count)] + self.eps * torch.randn(
                        (unused_count, self.data_dim), device=self.device).clone()

                print(f'************* Replaced ' + str(unused_count) + f' for codebook {k+1} *************')
                self.codebooks_used[k, :] = 0.0

    def encode(self, input_data):
        indices = []
        remainder = input_data

        for i in range(self.num_stages):
            _, _, min_indices = self.hard_vq(remainder, self.codebooks[i])
            indices.append(min_indices)
            remainder = remainder - self.codebooks[i][min_indices]

        return torch.stack(indices, dim=1)

    def decode(self, indices):
        reconstructed = torch.zeros((indices.shape[0], self.data_dim), device=self.device)
        for i in range(self.num_stages):
            reconstructed += self.codebooks[i][indices[:, i]]
        return reconstructed

    def compress_embeddings(self, embeddings, batch_size=64):
        compressed_embeddings = []
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Compressing embeddings"):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                indices = self.encode(batch)
            compressed_embeddings.append(indices.cpu().numpy())
        
        return np.vstack(compressed_embeddings)

    def search_similar(self, query_embedding, annoy_index, n_results=10):
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            compressed_query = self.encode(query_tensor).cpu().numpy().flatten()
        
        similar_indices = annoy_index.get_nns_by_vector(compressed_query, n_results)
        return similar_indices