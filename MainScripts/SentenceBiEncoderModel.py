from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceBiEncoder:
    def __init__(self, model_name = 'dunzhang/stella_en_400M_v5'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    
    def EncodeEmbeddings(self, chunks):
        return self.model.encode(chunks)
    
    def EncodeQueries(self, q):
        q = self.model.encode(q, prompt_name="s2p_query")
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        norm_q = q / norm
        return norm_q.tolist()