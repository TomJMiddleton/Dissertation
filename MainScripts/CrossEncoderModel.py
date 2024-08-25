from sentence_transformers import CrossEncoder
import torch

class MyCrossEncoder:
    def __init__(self, model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, default_activation_function=torch.nn.Sigmoid())
        self.model.model = self.model.model.to(self.device)
    
    def ReRankDocuments(self, query, retrieved_documents, top_n=10, window_size=512, overlap=128):
        """
        Re-rank the retrieved documents based on their relevance to the query using a sliding window approach.

        :param query: The query string
        :param retrieved_documents: A list of tuples (DocID, Title, CleanedDocument)
        :param top_n: Number of top documents to return after re-ranking
        :param window_size: The maximum token length for each window (chunk)
        :param overlap: The overlap size between consecutive windows
        :return: A list of tuples (DocID, Title, CleanedDocument, score) sorted by score in descending order
        """
        doc_score_pairs = []

        for doc in retrieved_documents:
            doc_id, title, cleaned_document = doc

            # Tokenize the document
            tokenized_doc = self.model.tokenizer.tokenize(cleaned_document)

            # Calculate the number of tokens available for the document after considering the query
            query_length = len(self.model.tokenizer.tokenize(query))
            available_length = window_size - query_length - 6

            # If the document is shorter than the available length, score it as is
            if len(tokenized_doc) <= available_length:
                query_document_pair = [query, cleaned_document]
                score = self.model.predict([query_document_pair])[0]
                doc_score_pairs.append((doc_id, title, cleaned_document, score))
            else:
                # Sliding window with overlap
                max_score = float('-inf')
                for i in range(0, len(tokenized_doc), available_length - overlap):
                    window = tokenized_doc[i:i + available_length]
                    window_text = self.model.tokenizer.convert_tokens_to_string(window)
                    query_document_pair = [query, window_text]
                    score = self.model.predict([query_document_pair])[0]
                    max_score = max(max_score, score)
                    if i + available_length >= len(tokenized_doc):
                        break

                # Store the document with its maximum score
                doc_score_pairs.append((doc_id, title, cleaned_document, max_score))

        # Sort the list by score in descending order
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[3], reverse=True)

        return doc_score_pairs[:top_n]


# ----Testing System----
if __name__ == "__main__":
    CE_model = MyCrossEncoder()
    print(CE_model.model.tokenizer.model_max_length)