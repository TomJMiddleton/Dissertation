import sqlite3
import os
from torch.utils.data import Dataset, DataLoader
from BERTNERSystem import NERModel
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class SQLiteDataset(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.exists(db_path), f"Could not locate the database at {db_path}"
        self.conn = None
        self.cur = None

    def _ensure_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cur = self.conn.cursor()

    def __len__(self):
        self._ensure_connection()
        self.cur.execute("SELECT COUNT(*) FROM Documents") 
        return self.cur.fetchone()[0]

    def __getitem__(self, idx):
        self._ensure_connection()
        self.cur.execute("SELECT DocID, SentenceText FROM Sentences LIMIT 1 OFFSET ?", (idx,))
        row = self.cur.fetchone()
        return row[0], row[1] 

    def __del__(self):
        if self.conn:
            self.conn.close()

class SQLiteDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def insert_keywords(self, entry_data):
        try:
            # Start a transaction
            self.connect()
            cur = self.conn.cursor()
            self.conn.execute('BEGIN')
            
            for doc_id, original_keyword, normalized_keyword in entry_data:
                # Check if the normalized keyword already exists in the Keywords table
                cur.execute("SELECT KeywordID FROM Keywords WHERE NormalizedKeyword = ?", (normalized_keyword,))
                result = cur.fetchone()
                
                if result:
                    keyword_id = result[0]
                else:
                    # Insert the new normalized keyword
                    cur.execute("INSERT INTO Keywords (NormalizedKeyword) VALUES (?)", (normalized_keyword,))
                    keyword_id = cur.lastrowid
                
                # Check if the original keyword variation already exists
                cur.execute("SELECT VariationID FROM KeywordVariations WHERE OriginalKeyword = ? AND KeywordID = ?", 
                                (original_keyword, keyword_id))
                if not cur.fetchone():
                    # Insert the new keyword variation
                    cur.execute("INSERT INTO KeywordVariations (KeywordID, OriginalKeyword) VALUES (?, ?)", 
                                    (keyword_id, original_keyword))
                
                # Check if the document-keyword relationship already exists
                cur.execute("SELECT * FROM DocumentKeywords WHERE DocID = ? AND KeywordID = ?", (doc_id, keyword_id))
                if not cur.fetchone():
                    # Insert the new document-keyword relationship
                    cur.execute("INSERT INTO DocumentKeywords (DocID, KeywordID) VALUES (?, ?)", (doc_id, keyword_id))
            
            # Commit the transaction
            self.conn.commit()
        except Exception as e:
            # Rollback in case of error
            self.conn.rollback()
            print(f"An error occurred: {e}")


def EntityNormalisation(entity):
    # Lowercase, remove punctuation
    entity = re.sub(r'[^\w\s]', '', entity.lower())
    
    # Lemmatize words
    if len(entity) <= 2:
        return entity
    lemmatizer = WordNetLemmatizer()
    entities = entity.split()
    ents = [lemmatizer.lemmatize(ent) for ent in entities]
    root_entity = ' '.join(ents)
    return root_entity

def process_database(db_path, model_name, batch_size=32):
    # Needed for stopwords and lemmatizer
    nltk.download('wordnet')
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

    # Initialize the NER model
    ner_model = NERModel(model_name, batch_size)

    # Create DataLoader
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

    # Initialize SQLiteDatabase for writing
    db = SQLiteDatabase(db_path)
    db.connect()

    # Process batches
    for batch in dataloader:
        doc_ids, texts = batch
        ner_results = ner_model.process_batch(list(texts))
        #print("----------")
        #print(ner_results)
        # Prepare keywords for insertion
        keywords = []
        for doc_id, result in zip(doc_ids, ner_results):
            for entity in result:
                original_keyword = entity['word']
                normalized_keyword = EntityNormalisation(original_keyword)
                if len(normalized_keyword) < 2 or normalized_keyword in STOPWORDS:
                    continue
                keywords.append((int(doc_id), original_keyword, normalized_keyword))
        #print("----------")
        #print(keywords)

        # Insert keywords into the database
        db.insert_keywords(keywords)

    # Disconnect from the database
    db.disconnect()

    print("NER processing complete. Results written to the database.")

# Usage
if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB2.db'
    model_name = 'huggingface-course/bert-finetuned-ner'
    batch_size = 64

    process_database(db_path, model_name, batch_size)