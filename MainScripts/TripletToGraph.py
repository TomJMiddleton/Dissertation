import csv
import sqlite3
import networkx as nx
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import regex as re
from fuzzywuzzy import fuzz
from tqdm import tqdm
from collections import defaultdict

class SQLKeyword:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.lemmatizer = WordNetLemmatizer()
        self.graph = nx.Graph()
        self.keyword_cache = {}
        self.fuzzy_match_cache = {}
        self.keyword_id_counter = 1

    def DBConn(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.load_existing_keywords()

    def DBDisconn(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
            self.cursor = None

    def load_existing_keywords(self):
        self.cursor.execute("SELECT KeywordID, NormalizedKeyword, RepresentativeKeyword FROM Keywords")
        for keyword_id, normalized_keyword, representative_keyword in self.cursor.fetchall():
            self.keyword_cache[normalized_keyword] = (keyword_id, representative_keyword)
            if keyword_id >= self.keyword_id_counter:
                self.keyword_id_counter = keyword_id + 1

    def entity_normalisation(self, entity):
        entity = entity.replace('-', ' ')
        entity = re.sub(r'[^\w\s]', '', entity.lower())
        
        if len(entity) <= 2:
            return entity
        entities = entity.split()
        ents = [self.lemmatizer.lemmatize(ent) for ent in entities]
        return ' '.join(ents)

    def fuzzy_match(self, normalized_entity, threshold=80):
        if normalized_entity in self.fuzzy_match_cache:
            return self.fuzzy_match_cache[normalized_entity]
        
        best_match = None
        best_score = 0
        
        for normalized_keyword, (keyword_id, representative_keyword) in self.keyword_cache.items():
            score = fuzz.token_sort_ratio(normalized_entity, normalized_keyword)
            if score > threshold and score > best_score:
                best_match = (keyword_id, representative_keyword)
                best_score = score
        
        self.fuzzy_match_cache[normalized_entity] = best_match
        return best_match

    def get_or_add_keyword(self, entity):
        normalized_entity = self.entity_normalisation(entity)
        
        if normalized_entity in self.keyword_cache:
            keyword_id, representative_keyword = self.keyword_cache[normalized_entity]
        else:
            match = self.fuzzy_match(normalized_entity)
            
            if match:
                keyword_id, representative_keyword = match
            else:
                keyword_id = self.keyword_id_counter
                self.keyword_id_counter += 1
                representative_keyword = entity
                self.keyword_cache[normalized_entity] = (keyword_id, representative_keyword)
                self.cursor.execute("""
                INSERT INTO Keywords (KeywordID, NormalizedKeyword, RepresentativeKeyword)
                VALUES (?, ?, ?)
                """, (keyword_id, normalized_entity, entity))

        # Add node to the graph
        self.graph.add_node(keyword_id, name=representative_keyword)
        
        return keyword_id

    def add_document_keyword(self, doc_id, keyword_id):
        self.cursor.execute("INSERT OR IGNORE INTO DocumentKeywords (DocID, KeywordID) VALUES (?, ?)", (doc_id, keyword_id))

    def process_document(self, doc_rows):
        edge_count = defaultdict(int)
        edge_set = set()

        for row in doc_rows:
            obj, relation, subject, confidence, doc_id = row
            confidence = float(confidence)
            
            object_id = self.get_or_add_keyword(obj)
            subject_id = self.get_or_add_keyword(subject)
            
            self.add_document_keyword(doc_id, object_id)
            self.add_document_keyword(doc_id, subject_id)

            edge_key = (object_id, relation)
            if edge_key not in edge_set:
                if edge_count[object_id] < 5 or confidence > 0.23:
                    if self.graph.has_edge(object_id, subject_id):
                        existing_confidence = self.graph[object_id][subject_id].get('confidence', 0)
                        if confidence > existing_confidence:
                            self.graph[object_id][subject_id]['relation'] = relation
                            self.graph[object_id][subject_id]['confidence'] = confidence
                    else:
                        self.graph.add_edge(object_id, subject_id, relation=relation, confidence=confidence)
                    
                    edge_count[object_id] += 1
                    edge_set.add(edge_key)

    def process_csv(self, csv_file):
        self.DBConn()
        total_rows = 18828
        
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            
            current_doc_id = None
            current_doc_rows = []
            pbar = tqdm(total=total_rows, desc="Processing CSV", unit="doc")
            for row in csv_reader:
                if float(row[3]) < 0.17: continue

                doc_id = row[4]
                
                if current_doc_id is None:
                    current_doc_id = doc_id
                
                if doc_id != current_doc_id:
                    self.process_document(current_doc_rows)
                    current_doc_rows = []
                    current_doc_id = doc_id
                    pbar.update(1)
                
                current_doc_rows.append(row)
                

            if current_doc_rows:
                self.process_document(current_doc_rows)
            pbar.close()
        
        self.DBDisconn()

    def export_graph(self, file_path):
        nx.write_graphml(self.graph, file_path)
        print(f"Graph exported to {file_path}")

    def print_graph_summary(self):
        print(f"Graph Summary:")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print("Sample of nodes:")
        for node in list(self.graph.nodes(data=True))[:5]:
            print(node)
        print("Sample of edges:")
        for edge in list(self.graph.edges(data=True))[:5]:
            print(edge)

if __name__ == "__main__":
    csv_file = './Datasets/Processed/20NG/NGtriplets.csv'
    db_file = './Datasets/FinalDB/FinalSQLDB.db'
    graph_file = './Datasets/FinalDB/NGKG.graphml'

    keyword_db = SQLKeyword(db_file)
    keyword_db.process_csv(csv_file)
    keyword_db.print_graph_summary()
    keyword_db.export_graph(graph_file)