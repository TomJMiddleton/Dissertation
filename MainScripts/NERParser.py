import sqlite3
import os
from torch.utils.data import Dataset, DataLoader
from BERTNERSystem import NERModel

class SQLiteDataset(Dataset):
    def __init__(self, db_path, table_name='Sentence', text_column='sent'):
        self.db_path = db_path
        assert os.path.exists(db_path), f" Could not locate the database at {db_path}"
        self.table_name = table_name
        self.text_column = text_column
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        self.len = self.cur.fetchone()[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        self.cur.execute(f"SELECT SentID, {self.text_column} FROM {self.table_name} LIMIT 1 OFFSET {idx}")
        row = self.cur.fetchone()
        return row[0], row[1]  # return SentID and sent

    def __del__(self):
        self.conn.close()

class SQLiteDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cur = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def insert_keywords(self, keywords):
        self.cur.executemany('''INSERT INTO Keyword 
                                (SentID, keyword) 
                                VALUES (?, ?)''', keywords)

    def commit(self):
        self.conn.commit()

def process_database(db_path, model_name, batch_size=32):
    # Initialize the NER model
    ner_model = NERModel(model_name, batch_size)

    # Create DataLoader
    dataset = SQLiteDataset(db_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    # Initialize SQLiteDatabase for writing
    db = SQLiteDatabase(db_path)
    db.connect()

    # Process batches
    for batch in dataloader:
        sent_ids, texts = batch
        ner_results = ner_model.process_batch(list(texts))
        #print("-----------")
        #print(ner_results)
        # Prepare keywords for insertion
        keywords = []
        for sent_id, result in zip(sent_ids, ner_results):
            for entity in result:
                #print("-----------")
                #print((int(sent_id), entity['word']))
                keywords.append((int(sent_id), entity['word']))

        # Insert keywords into the Keyword table
        db.insert_keywords(keywords)

        # Commit after each batch
        db.commit()

    # Disconnect from the database
    db.disconnect()

    print(f"NER processing complete. Results written to Keyword table")

# Usage
if __name__ == "__main__":
    db_path = './Datasets/Database/NewsGroupDB.db'
    input_table = 'Sentence'
    input_column = 'sent'
    output_table = 'Keyword'
    model_name = 'huggingface-course/bert-finetuned-ner'
    batch_size = 32  # Adjust based on your GPU memory and performance requirements

    process_database(db_path, model_name, batch_size)