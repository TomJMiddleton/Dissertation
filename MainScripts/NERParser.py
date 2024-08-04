import sqlite3
import os
from torch.utils.data import Dataset, DataLoader
from BERTNERSystem import NERModel
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import time, sys
from transformers import AutoTokenizer
import torch
import warnings

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
        self.cur.execute("SELECT DocID, CleanedDocument FROM Documents WHERE DocID = ?", (idx + 1,))
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
    entity = entity.replace('-', ' ')
    entity = re.sub(r'[^\w\s]', '', entity.lower())
    
    # Lemmatize words
    if len(entity) <= 2:
        return entity
    lemmatizer = WordNetLemmatizer()
    entities = entity.split()
    ents = [lemmatizer.lemmatize(ent) for ent in entities]
    root_entity = ' '.join(ents)
    return root_entity

def split_text(text, tokenizer, max_length=500, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    
    return chunks

def process_database(db_path, model_name, batch_size=32, n_workers = 4):
    # Needed for stopwords and lemmatizer
    nltk.download('wordnet')
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

    # Initialize the NER model
    ner_model = NERModel(model_name, batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create DataLoader
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)

    # Initialize SQLiteDatabase for writing
    db = SQLiteDatabase(db_path)
    db.connect()

    start_time = time.time()
    print(" User note: Max token limit warning message is handled")
    # Process batches
    for batch in dataloader:
        doc_ids, texts = batch

        # Split long texts
        split_texts = []
        split_doc_ids = []
        for doc_id, text in zip(doc_ids, texts):
            chunks = split_text(text, tokenizer)
            split_texts.extend(chunks)
            split_doc_ids.extend([doc_id] * len(chunks))
        
        # Adjust batch size if necessary
        ner_batch_size = min(batch_size, len(split_texts))

        # Inference on batch
        ner_results = ner_model.process_batch(list(split_texts), ner_batch_size)

        # Combine results for split documents
        combined_results = {}
        for doc_id, result in zip(split_doc_ids, ner_results):
            if doc_id not in combined_results:
                combined_results[doc_id] = []
            combined_results[doc_id].extend(result)

        #print("----------")
        #print(ner_results)
        # Prepare keywords for insertion
        keywords = []
        for doc_id, result in combined_results.items():
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
        #k_out = keywords

    # Disconnect from the database
    db.disconnect()
    #print(k_out)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f" Execution time: {execution_time} seconds \n N-Workers = {n_workers}")

# Usage
if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB2.db'
    model_name = 'huggingface-course/bert-finetuned-ner'
    batch_size = 64

    #print(torch.__version__)
    #print(torch.cuda.is_available())
    #print(torch.backends.cuda.flash_sdp_enabled())
    #print(torch.version.cuda)
    process_database(db_path, model_name, batch_size)
    