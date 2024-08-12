from sentence_transformers import CrossEncoder

class MyCrossEncoder:
    def __init__(self, model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
    
    def ReRankDocuments(self, query, retrieved_documents, top_n=10):
        """
        Re-rank the retrieved documents based on their relevance to the query.

        :param query: The query string
        :param retrieved_documents: A list of retrieved document strings
        :param top_n: Number of top documents to return after re-ranking
        :return: A list of tuples (document, score) sorted by score in descending order
        """
        # Query, Doc pairs
        query_document_pairs = [[query, doc] for doc in retrieved_documents]
        
        # Relevance scores 
        scores = self.model.predict(query_document_pairs)
        
        # Doc, score pairs
        doc_score_pairs = list(zip(retrieved_documents, scores))
        
        # Sort by score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_n]
    