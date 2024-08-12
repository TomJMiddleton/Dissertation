from NERParser import SQLiteDatabase
from SentenceBiEncoderModel import SentenceBiEncoder
from annoy import AnnoyIndex
import annoy
import numpy as np


def SearchIndex(index, qv, n=20):
    return index.get_nns_by_vector(qv, n, include_distances=False)

def LoadIndex(dim = 1024, index_path = './Datasets/Database/NGAnnoyVec.ann', prefault_mode = False):
    index = AnnoyIndex(dim, 'angular')
    index.load(index_path, prefault=prefault_mode)
    return index

if __name__ == "__main__":
    # Testing System
    print("--------- \n Reading INDEX and instantiating MODEL \n ----------------")
    index = LoadIndex()
    bi_encoder = SentenceBiEncoder()

    print("--------- \n ENCODING QUERY EMBEDDINGS \n ----------------")
    query = "I want to learn more about cars. How much would a nice car cost?"
    query_vector = bi_encoder.EncodeQuery([query])[0].tolist()

    print("--------- \n SEARCHING INDEX \n ----------------")
    indices = SearchIndex(index, query_vector)

    print(f"Query: {query}")
    print("Results:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. Index: {idx}")
    
    db = SQLiteDatabase('./Datasets/Database/NewsGroupDB3.db')

    doc_ids = [idx + 1 for idx in indices]
    query_results = db.RetrieveDocFromID(doc_ids)
    for doc in query_results:
        docid, title, cleaned_document = doc
        print(f"\n ---------------- \n Title: {title} \n {cleaned_document[:500]} \n ----------------")