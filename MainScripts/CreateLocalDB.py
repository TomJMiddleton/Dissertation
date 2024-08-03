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
            
            # Create Documents table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS Documents (
                DocID INTEGER PRIMARY KEY,
                Title TEXT NOT NULL,
                CleanedDocument TEXT
            )
            ''')

            # Create Keywords table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS Keywords (
                KeywordID INTEGER PRIMARY KEY,
                NormalizedKeyword TEXT NOT NULL UNIQUE
            )
            ''')

            # Create KeywordVariations table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS KeywordVariations (
                VariationID INTEGER PRIMARY KEY,
                KeywordID INTEGER,
                OriginalKeyword TEXT NOT NULL,
                FOREIGN KEY (KeywordID) REFERENCES Keywords(KeywordID)
            )
            ''')

            # Create DocumentKeywords table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS DocumentKeywords (
                DocID INTEGER,
                KeywordID INTEGER,
                PRIMARY KEY (DocID, KeywordID),
                FOREIGN KEY (DocID) REFERENCES Documents(DocID),
                FOREIGN KEY (KeywordID) REFERENCES Keywords(KeywordID)
            )
            ''')

            # Create Sentences table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS Sentences (
                SentenceID INTEGER PRIMARY KEY,
                DocID INTEGER,
                SentenceText TEXT NOT NULL,
                FOREIGN KEY (DocID) REFERENCES Documents(DocID)
            )
            ''')

            # Create DocumentSimilarities table
            cur.execute('''
            CREATE TABLE IF NOT EXISTS DocumentSimilarities (
                DocID1 INTEGER,
                DocID2 INTEGER,
                SimilarityScore REAL NOT NULL,
                PRIMARY KEY (DocID1, DocID2),
                FOREIGN KEY (DocID1) REFERENCES Documents(DocID),
                FOREIGN KEY (DocID2) REFERENCES Documents(DocID)
            )
            ''')

            # Create indexes for improved query performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_normalized_keyword ON Keywords(NormalizedKeyword)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_original_keyword ON KeywordVariations(OriginalKeyword)')

    return True

if __name__ == "__main__":
    file_path = './Datasets/Database/testingDB.db'
    if InstatiateDB(file_path):
        print("Database created successfully.")