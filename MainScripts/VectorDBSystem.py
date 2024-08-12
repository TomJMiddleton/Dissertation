from VecSearch import LoadIndex, SearchIndex
from SentenceBiEncoderModel import SentenceBiEncoder
from CrossEncoderModel import MyCrossEncoder
from NERParser import SQLiteDatabase

def FullVectorSystem(bi_encoder_model, cross_encoder_model, index, db_io, queries_text, n_rank, n_rerank):
    # bi-encoder generates query embeddings
    query_vectors = bi_encoder_model.EncodeQueries(queries_text)

    queries_results = []
    for q_vec, q_text in zip(query_vectors, queries_text):
        ranked_indices = SearchIndex(index, q_vec, n=n_rank)
        ranked_doc_ids = [idx + 1 for idx in ranked_indices]
        ranked_docs = db_io.RetrieveDocFromID(ranked_doc_ids)
        reranked_docs = cross_encoder_model.ReRankDocuments(q_text, ranked_docs, top_n=n_rerank)
        queries_results.append(reranked_docs)

    return zip(queries_text, queries_results)

def print_re_ranked_results(zipped_results):
    for query, results in zipped_results:
        print(f"Query: {query}")
        print("=" * (len(query) + 7))

        for rank, (doc_id, title, cleaned_document, score) in enumerate(results, 1):
            print(f"Rank: {rank}")
            print(f"Document ID: {doc_id}")
            print(f"Title: {title}")
            print(f"Relevance Score: {score:.4f}")
            print(f"Snippet: {cleaned_document[:1000]}") 
            print("-" * 50)
        
        print("\n" + "=" * 50 + "\n")




# ----Testing System----
if __name__ == "__main__":
    # ----Instantiate key systems----
    print("---------------- \n Reading INDEX and Instantiating MODELS \n ----------------")
    index = LoadIndex(dim = 1024, index_path = './Datasets/Database/NGAnnoyVec.ann', prefault_mode = False)
    bi_encoder = SentenceBiEncoder()
    cross_encoder = MyCrossEncoder()
    db = SQLiteDatabase('./Datasets/Database/NewsGroupDB3.db')

    # ----Define Queries----
    query_one = "What is JPEG and why do we use that file format?"
    query_two = "Who is the best first baseman in Major League Baseball?"
    query_three = "Do you have any information on Mike Harpe?"
    query_four = "I have cholistatis, what diet should I be on?"
    queries = [query_one, query_two, query_three, query_four]

    # ----Define Search Parameters----
    n_rank = 80
    n_rerank = 8

    # Test Master Vec search fn.
    results = FullVectorSystem(bi_encoder, cross_encoder, index, db, queries, n_rank, n_rerank)
    print_re_ranked_results(results)
