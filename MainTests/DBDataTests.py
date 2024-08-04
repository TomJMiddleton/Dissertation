import sqlite3, csv
from contextlib import closing


# Export n sentences from the Sentence table to a text file
def ReviewSentenceTableInTxtFile(db_path, output_file, n):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT sent FROM Sentence LIMIT ?", (n,))
            sentences = cur.fetchall()
            with open(output_file, 'w', encoding='utf-8') as f:
                for sentence in sentences:
                    f.write(sentence[0] + '\n')
    print(f"{len(sentences)} sentences have been written to {output_file}")


# NG sentence processing visual check
"""
ReviewSentenceTableInTxtFile(db_path='./Datasets/Database/NewsGroupDB.db',
                             output_file = 'db_export_sentences.txt',
                             n = 3000)
"""

# Export n entities from the Keyword table to a text file
def ReviewSentenceTableInTxtFile(db_path, output_file, n):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT DocID, SentenceText FROM Sentences LIMIT ?", (n,))
            entities = cur.fetchall()
            with open(output_file, 'w', encoding='utf-8') as f:
                for entity in entities:
                    f.write(str(entity) + '\n')
    print(f"{len(entities)} sentences have been written to {output_file}")


# NG entity processing visual check
"""
ReviewSentenceTableInTxtFile(db_path='./Datasets/Database/NewsGroupDB3.db',
                             output_file = 'db3_export_entities.txt',
                             n = 3000)
"""

def CheckSentValue(db_path, sent_id):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT sent FROM Sentence WHERE SentID = ?", (sent_id,))
            result = cur.fetchone()
            if result:
                print(f"Sentence with SentID {sent_id}:")
                print(result[0])
            else:
                print(f"No sentence found with SentID {sent_id}")

#CheckSentValue(db_path='./Datasets/Database/NewsGroupDB.db', sent_id = 14343)

def CheckDocValue(db_path, doc_id):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT document FROM Document WHERE DocID = ?", (doc_id,))
            result = cur.fetchone()
            if result:
                print(f"Sentence with DocID {doc_id}:")
                print(result[0])
            else:
                print(f"No sentence found with DocID {doc_id}")

#CheckDocValue(db_path='./Datasets/Database/NewsGroupDB.db', doc_id = 14343)

def GetTableCount(db_path, table_name):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            print(row_count)

#GetTableCount(db_path='./Datasets/Database/NewsGroupDB.db', table_name = 'Keyword')


def TableToCSV(db_path, table_name, fields):
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.cursor()) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            print(f"Number of rows in the {table_name} table: {row_count}")

            cur.execute(f"SELECT {', '.join(fields)} FROM {table_name}")
            rows = cur.fetchall()
            
            with open(f'./Datasets/Database/{table_name}.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields) 
                csvwriter.writerows(rows)  


TableToCSV('./Datasets/Database/NewsGroupDB3.db', 'Keywords', ['KeywordID', 'NormalizedKeyword'])
TableToCSV('./Datasets/Database/NewsGroupDB3.db', 'KeywordVariations', ['VariationID', 'KeywordID', 'OriginalKeyword'])
TableToCSV('./Datasets/Database/NewsGroupDB3.db', 'DocumentKeywords', ['DocID', 'KeywordID'])
#TableToCSV('./Datasets/Database/NewsGroupDB3.db', 'Documents', ['DocID', 'Title', 'CleanedDocument'])
#TableToCSV('./Datasets/Database/NewsGroupDB3.db', 'Sentences', ['SentenceID', 'DocID', 'SentenceText'])
#TableToCSV('./Datasets/Database/NewsGroupDB2.db', 'DocumentKeywords', ['DocID1', 'DocID2', 'SimilarityScore'])
