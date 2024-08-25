import torch, logging, sqlite3, time, string, textwrap
import numpy as np
from typing import List, Dict, Tuple
from SentenceBiEncoderModel import SentenceBiEncoder
from CrossEncoderModel import MyCrossEncoder
from RVQModel import RVQ
from annoy import AnnoyIndex
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from adjustText import adjust_text

class NLPSearchSystem:
    def __init__(self, rvq_model_path: str, rvq_annoy_index_path: str, original_annoy_index_path: str,
                 original_embeddings_path: str, num_stages: int, vq_bitrate_per_stage: int, data_dim: int,
                 db_path: str, kg_path: str, log_file: str = 'gpu_memory_usage.log') -> None:
        self.logger = self._setup_logger(log_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_path = db_path

        # Set up Logging for GPU Memory
        self._log_memory_usage("Initial")
        
        # Initialize models
        self.bi_encoder = SentenceBiEncoder()
        self._log_memory_usage("After loading bi-encoder")
        
        self.cross_encoder = MyCrossEncoder()
        self._log_memory_usage("After loading cross-encoder")
        
        self.rvq_model = self._load_rvq_model(rvq_model_path, num_stages, vq_bitrate_per_stage, data_dim)
        self._log_memory_usage("After loading RVQ model")
        
        # Load Annoy indices
        self.rvq_annoy_index = self._load_annoy_index(rvq_annoy_index_path, num_stages, 'hamming')
        self.original_annoy_index = self._load_annoy_index(original_annoy_index_path, data_dim, 'angular')
        self._log_memory_usage("After loading Annoy indices")
        
        # Load original embeddings
        self.original_embeddings = np.load(original_embeddings_path)
        self._log_memory_usage("After loading embeddings")

        # Load Knowledge Graph
        self.kg = nx.read_graphml(kg_path)
        self._log_memory_usage("After loading Knowledge Graph")

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Initialize the database connection
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()

        # Initialize fuzzy match cache
        self.fuzzy_match_cache = {}

    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger('GPUMemoryLogger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
        
    def _log_memory_usage(self, step: str):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            log_message = f"Memory Usage {step}: {memory_allocated:.2f} MB (allocated), {memory_reserved:.2f} MB (reserved)"
            self.logger.info(log_message)
    
    def _load_rvq_model(self, model_path: str, num_stages: int, vq_bitrate_per_stage: int, data_dim: int) -> RVQ:
        rvq_model = RVQ(num_stages=num_stages, 
                        vq_bitrate_per_stage=vq_bitrate_per_stage, 
                        data_dim=data_dim, 
                        device=self.device)
        
        state_dict = torch.load(model_path, map_location=self.device)
        rvq_model.load_state_dict(state_dict)
        rvq_model.eval()
        
        return rvq_model
    
    def _load_annoy_index(self, index_path: str, dim: int, metric: str, prefault_mode: bool = False) -> AnnoyIndex:
        index = AnnoyIndex(dim, metric)
        index.load(index_path, prefault=prefault_mode)
        return index

    def entity_normalisation(self, entity):
        entity = entity.replace('-', ' ')
        entity = re.sub(r'[^\w\s]', '', entity.lower())
        
        if len(entity) <= 2:
            return entity
        entities = entity.split()
        ents = [self.lemmatizer.lemmatize(ent) for ent in entities]
        return ' '.join(ents)

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def fuzzy_match(self, normalized_entity, threshold=80):
        if normalized_entity in self.fuzzy_match_cache:
            return self.fuzzy_match_cache[normalized_entity]
        
        # Perform a database search for similar keywords
        self.cur.execute("""
            SELECT NormalizedKeyword, KeywordID
            FROM Keywords
            WHERE NormalizedKeyword LIKE ?
        """, (f'%{normalized_entity}%',))
        
        potential_matches = self.cur.fetchall()
        
        best_match = None
        best_score = 0
        best_id = None
        
        for keyword, keyword_id in potential_matches:
            score = fuzz.token_sort_ratio(normalized_entity, keyword)
            if score > threshold and score > best_score:
                best_match = keyword
                best_score = score
                best_id = keyword_id
        
        self.fuzzy_match_cache[normalized_entity] = (best_match, best_id)
        return best_match, best_id

    def search_kg_by_key_terms(self, query, max_depth=2, threshold=80):
        # Extract key terms from the query
        key_terms = self.preprocess_text(query)
        
        # Find keywords that match the key terms
        matched_nodes = set()
        for term in key_terms:
            normalized_term = self.entity_normalisation(term)
            fuzzy_match, keyword_id = self.fuzzy_match(normalized_term, threshold)
            if fuzzy_match:
                matched_nodes.add(str(keyword_id)) 
        
        # Perform a breadth-first search from matched nodes
        subgraph_nodes = set(matched_nodes)
        for start_node in matched_nodes:
            neighbors = set()
            current_depth_nodes = {start_node}
            for _ in range(max_depth):
                next_depth_nodes = set()
                for node in current_depth_nodes:
                    next_depth_nodes.update(self.kg.neighbors(node))
                neighbors.update(next_depth_nodes)
                current_depth_nodes = next_depth_nodes
            subgraph_nodes.update(neighbors)
        
        # Remove self-loops
        frozen_subgraph = self.kg.subgraph(subgraph_nodes)
        subgraph = nx.Graph(frozen_subgraph)
        subgraph.remove_edges_from(nx.selfloop_edges(subgraph))
        
        # Visualize the subgraph
        fig = self.visualize_subgraph(subgraph, matched_nodes)
        
        # Prepare results
        results = []
        for node in subgraph.nodes():
            neighbors = list(subgraph.neighbors(node))
            edges = [(node, neighbor, subgraph[node][neighbor]['relation']) for neighbor in neighbors]
            results.append({
                'node': subgraph.nodes[node]['name'],
                'matched': node in matched_nodes,
                'edges': edges
            })
        
        return results, fig

    def visualize_subgraph(self, subgraph: nx.Graph, matched_nodes: set, max_nodes: int = 40) -> plt.Figure:
        # Ensure matched_nodes are actually in the subgraph
        matched_nodes = set(matched_nodes) & set(subgraph.nodes())


        if len(subgraph) > max_nodes:
            important_nodes = list(matched_nodes) + [node for node, _ in sorted(subgraph.degree, key=lambda x: x[1], reverse=True) 
                                                    if node not in matched_nodes][:max_nodes-len(matched_nodes)]
            subgraph = subgraph.subgraph(important_nodes)
        
        plt.figure(figsize=(20, 16))
        
        # different layout algorithms
        #pos = nx.spring_layout(subgraph, k=1, iterations=50)  # Increase k for more spread
        #pos = nx.kamada_kawai_layout(subgraph)  # Alternative layout algorithm
        pos = nx.fruchterman_reingold_layout(subgraph, k=0.5, iterations=50)  # Another alternative

        # Node sizes based on degree, but with a smaller range
        max_degree = max(dict(subgraph.degree()).values())
        node_sizes = {node: 300 + 200 * (subgraph.degree(node) / max_degree) for node in subgraph.nodes()}

        non_matched_nodes = set(subgraph.nodes()) - matched_nodes
        nx.draw_networkx_nodes(subgraph, pos, nodelist=list(non_matched_nodes), 
                            node_color='lightblue', node_size=[node_sizes[node] for node in non_matched_nodes])

        nx.draw_networkx_nodes(subgraph, pos, nodelist=list(matched_nodes), 
                            node_color='lightgreen', node_size=[node_sizes[node] * 1.5 for node in matched_nodes])

        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.3, arrows=True, arrowsize=10)

        labels = {node: subgraph.nodes[node]['name'] for node in subgraph.nodes()}

        texts = []
        for node, label in labels.items():
            x, y = pos[node]
            texts.append(plt.text(x, y, label, fontsize=8, ha='center', va='center'))

        
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), expand_points=(1.2, 1.2))

        edge_labels = {(u, v): d['relation'] for u, v, d in subgraph.edges(data=True) 
                    if u in matched_nodes or v in matched_nodes}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

        plt.title("Knowledge Graph Subgraph", fontsize=20)
        plt.axis('off')

        plt.legend([plt.Circle((0,0),1, facecolor='lightgreen'), 
                    plt.Circle((0,0),1, facecolor='lightblue')],
                ['Matched Nodes', 'Non-matched Nodes'],
                loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return plt.gcf() 

    def rank_kg_results(self, results, query):
        query_terms = set(self.preprocess_text(query))
        ranked_results = []
        
        for result in results:
            node_terms = set(self.preprocess_text(result['node']))
            relevance_score = len(query_terms.intersection(node_terms))
            
            if result['matched']:
                relevance_score += 5
            
            ranked_results.append((relevance_score, result))
        
        return [result for _, result in sorted(ranked_results, key=lambda x: x[0], reverse=True)]

    def search_kg(self, query: str, max_depth: int = 2, threshold: int = 80) -> Tuple[List[Dict], plt.Figure]:
        results, fig = self.search_kg_by_key_terms(query, max_depth, threshold)
        ranked_results = self.rank_kg_results(results, query)
        return ranked_results, fig
    
    def _quantize_query(self, query_embedding: np.ndarray) -> np.ndarray:
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.rvq_model.encode(query_tensor).cpu().numpy().flatten()
        
    def _annoy_search(self, index: AnnoyIndex, query_vector: np.ndarray, k: int) -> List[int]:
        return index.get_nns_by_vector(query_vector, k)
        
    def _compute_similarities(self, query_embedding: np.ndarray, retrieved_indices: List[int], k: int) -> List[Tuple[int, float]]:
        retrieved_embeddings = self.original_embeddings[retrieved_indices]
        similarities = np.dot(retrieved_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(retrieved_indices[i], similarities[i]) for i in top_k_indices]
        
    def _rerank_results(self, query: str, top_k_results: List[Tuple[int, float]], n: int) -> List[Dict]:
        unique_idx_set = {idx for idx, _ in top_k_results}
        unique_documents = [doc for doc in (self._get_document_content(idx) for idx in unique_idx_set) if doc is not None]
        reranked = self.cross_encoder.ReRankDocuments(query, unique_documents, top_n=min(n, len(unique_documents)))
        return reranked
        
    def _get_document_content(self, sentence_id: int) -> Tuple[int, str, str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                
                # Get the SentenceID, Title, and SentenceText in a single query
                cur.execute('''
                SELECT Sentences.SentenceID, Documents.Title, Sentences.SentenceText
                FROM Sentences
                JOIN Documents ON Sentences.DocID = Documents.DocID
                WHERE Sentences.SentenceID = ?
                ''', (sentence_id,))
                
                result = cur.fetchone()
                if result is None:
                    raise ValueError(f"No sentence found for SentenceID: {sentence_id}")
                
                return result  # This is a tuple of (SentenceID, Title, SentenceText)
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None
        except Exception as e:
            print(f"Error in _get_document_content: {e}")
            return None
            
    def _generate_answer(self, query: str, reranked_results: List[Dict], kg_results: List[Dict]) -> str:
        # Implement logic to generate a succinct answer or summary
        return ""

    def search(self, query: str, use_rvq: bool = True, RVQ_n: int = 200, bi_n: int = 50, cross_n: int = 5, kg_n: int = 10) -> Dict:
        self._log_memory_usage("Before search")
        
        # 1. Encode the query using bi-encoder
        query_embedding = self.bi_encoder.EncodeQueries([query])[0]
        self._log_memory_usage("After query encoding")
        
        # 2. Perform search based on the specified method
        if use_rvq:
            quantized_query = self._quantize_query(query_embedding)
            retrieved_indices = self._annoy_search(self.rvq_annoy_index, quantized_query, RVQ_n)
        else:
            retrieved_indices = self._annoy_search(self.original_annoy_index, query_embedding, RVQ_n)
        self._log_memory_usage("After Annoy search")
        
        # 3. Compute similarities 
        top_k_results = self._compute_similarities(query_embedding, retrieved_indices, bi_n)
        
        # 4. Re-rank using cross-encoder
        reranked_results = self._rerank_results(query, top_k_results, cross_n)
        self._log_memory_usage("After re-ranking")
        
        # 5. Search Knowledge Graph
        kg_results, kg_fig = self.search_kg(query)
        kg_results = kg_results[:kg_n]
        self._log_memory_usage("After KG search")
        
        # 6. Generate a succinct answer (placeholder)
        answer = self._generate_answer(query, reranked_results, kg_results)
        self._log_memory_usage("After generating answer")
        
        return {
            "answer": answer,
            "relevant_info": reranked_results,
            "kg_info": kg_results,
            "kg_visualization": kg_fig
        }

    def __del__(self):
        # Close the database connection when the object is destroyed
        self.conn.close()


def write_search_results_to_file(result, input_query, search_time, width=80):
    def wrap_text(text, width=width, initial_indent='', subsequent_indent=''):
        lines = text.splitlines()
        wrapped_lines = []
        for line in lines:
            wrapped = textwrap.fill(line, width=width, initial_indent=initial_indent,
                                    subsequent_indent=subsequent_indent,
                                    replace_whitespace=False, break_long_words=False)
            wrapped_lines.append(wrapped)
        return '\n'.join(wrapped_lines)

    filename = "SearchResult-" + input_query[:25] + ".txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\n" + wrap_text(f"Search completed in {search_time:.4f} seconds\n"))
        file.write("\n" + wrap_text(f"QUERY: {input_query}\n\n"))
        file.write("\n" + wrap_text(f"ANSWER: {result['answer']}\n\n"))
        
        file.write("Relevant Information:\n")
        for doc_trip in result["relevant_info"]:
            file.write("-" * width + "\n")
            file.write("\n" + wrap_text(f"DOC ID: {doc_trip[0]}\n"))
            file.write("\n" + wrap_text(f"DOC Title: {doc_trip[1]}\n"))
            file.write("\n" + wrap_text(f"CONFIDENCE: {doc_trip[3]}\n"))
            file.write("\n" + wrap_text(f"DOC TEXT: {doc_trip[2][:400]}...\n\n"))

        file.write("Knowledge Graph Results:\n")
        for i, kg_result in enumerate(result["kg_info"], 1):
            file.write("\n" + wrap_text(f"{i}. Node: {kg_result['node']}\n"))
            file.write("\n" + wrap_text(f"   Directly matched to query: {'Yes' if kg_result['matched'] else 'No'}\n"))
            file.write("   Edges:\n")
            for j, edge in enumerate(kg_result['edges'][:15], 1):
                target_node = search_system.kg.nodes[edge[1]]['name']
                file.write("\n" + wrap_text(f"     {j}. {target_node} ({edge[2]})\n", 
                                     initial_indent='', subsequent_indent='        '))
            if len(kg_result['edges']) > 15:
                file.write("\n" + wrap_text(f"     ... and {len(kg_result['edges']) - 15} more edges\n"))

        file.write("\n" + wrap_text(f"Search completed in {search_time:.4f} seconds"))

    print(f"Search results have been written to {filename}")

if __name__ == "__main__":
    search_system = NLPSearchSystem(
        rvq_model_path="./Datasets/FinalDB/RVQ_Model.pt",
        rvq_annoy_index_path="./Datasets/FinalDB/NGRVQVec.ann",
        original_annoy_index_path="./Datasets/FinalDB/NGAnnoyVec.ann",
        original_embeddings_path="./Datasets/FinalDB/.original_embeddings.npy",
        num_stages=4,
        vq_bitrate_per_stage=6,
        data_dim=1024,
        db_path="./Datasets/FinalDB/FinalSQLDB.db",
        kg_path="./Datasets/FinalDB/NGKG.graphml",
        log_file='gpu_memory_usage.log'
    )
    
    input_query = "What is JPEG and why do we use that file format?"
    start_time = time.time()
    result = search_system.search(input_query, use_rvq=False)
    end_time = time.time()
    search_time = end_time - start_time
    write_search_results_to_file(result, input_query, search_time, width=100)

    kg_fig = result["kg_visualization"]
    kg_fig.savefig("knowledge_graph_subgraph.png")
    print("\nKnowledge graph visualization saved as 'knowledge_graph_subgraph.png'")

    plt.show()  