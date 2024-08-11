from NERParser import SQLiteDatabase
from SentenceBiEncoderModel import SentenceBiEncoder
from annoy import AnnoyIndex
import annoy
import numpy as np


def search_index(index, qv, n=20):
    return index.get_nns_by_vector(qv, n, include_distances=False)

if __name__ == "__main__":
    # Testing System
    print("--------- \n Reading INDEX and instantiating MODEL \n ----------------")
    index = AnnoyIndex(1024, 'angular')
    index.load('./Datasets/Database/NGAnnoyVec.ann', prefault=False)
    bi_encoder = SentenceBiEncoder()

    print("--------- \n ENCODING QUERY EMBEDDINGS \n ----------------")
    query = "I want to learn more about cars. How much would a nice car cost?"
    query_vector = bi_encoder.EncodeQuery([query])[0].tolist()

    print("--------- \n SEARCHING INDEX \n ----------------")
    indices = search_index(index, query_vector)

    print(f"Query: {query}")
    print("Results:")
    for i, idx in enumerate(indices):
        print(f"{i+1}. Index: {idx}")
    
    db = SQLiteDatabase('./Datasets/Database/NewsGroupDB3.db')

    doc_ids = [idx + 1 for idx in indices]
    query_results = db.RetrieveDocFromID(doc_ids)
    for doc in query_results:
        title, cleaned_document = doc
        print(f"\n ---------------- \n Title: {title} \n {cleaned_document[:500]} \n ----------------")