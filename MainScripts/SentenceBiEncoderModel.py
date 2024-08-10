from sentence_transformers import SentenceTransformer

class SentenceBiEncoder:
    def __init__(self, model_name = 'dunzhang/stella_en_400M_v5'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    
    def EncodeEmbeddings(self, chunks):
        return self.model.encode(chunks)