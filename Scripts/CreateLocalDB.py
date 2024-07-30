import sqlite3
from contextlib import closing
import os

# Create the database
def InstatiateDB(file_path):
    if os.path.exists(file_path):
        print(f" Database already exists at {file_path}")
        return False
    with closing(sqlite3.connect(file_path)) as conn:
        with closing(conn.cursor()) as cur:
            # Document table 
            cur.execute('''CREATE TABLE IF NOT EXISTS Document
                        (DocID INTEGER PRIMARY KEY AUTOINCREMENT,
                            DocText TEXT)''')

            # Sentence table 
            cur.execute('''CREATE TABLE IF NOT EXISTS Sentence
                        (SentID INTEGER PRIMARY KEY AUTOINCREMENT,
                            DocID INTEGER,
                            sent TEXT,
                            FOREIGN KEY (DocID) REFERENCES Document(DocID))''')

            # Keyword table
            cur.execute('''CREATE TABLE IF NOT EXISTS Keyword
                        (KeywordID INTEGER PRIMARY KEY AUTOINCREMENT,
                            SentID INTEGER,
                            keyword TEXT,
                            FOREIGN KEY (SentID) REFERENCES Sentence(SentID))''')

            # Similarity table 
            cur.execute('''CREATE TABLE IF NOT EXISTS Similarity
                        (DocID1 INTEGER,
                            DocID2 INTEGER,
                            similarity REAL,
                            PRIMARY KEY (DocID1, DocID2),
                            FOREIGN KEY (DocID1) REFERENCES Document(DocID),
                            FOREIGN KEY (DocID2) REFERENCES Document(DocID))''')
    return True

if __name__ == "__main__":
    file_path = './Datasets/Database/MasterDB.db'
    if InstatiateDB(file_path):
        print("Database created successfully.")