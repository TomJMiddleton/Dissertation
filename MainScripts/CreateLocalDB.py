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
                NormalizedKeyword TEXT NOT NULL UNIQUE,
                RepresentativeKeyword TEXT NOT NULL
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

            # Create indexes for improved query performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_normalized_keyword ON Keywords(NormalizedKeyword)')

    return True

if __name__ == "__main__":
    file_path = './Datasets/FinalDB/FinalSQLDB.db'
    if InstatiateDB(file_path):
        print("Database created successfully.")