import sqlite3
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
ReviewSentenceTableInTxtFile(db_path='./Datasets/Database/NewsGroupDB.db',
                             output_file = 'db_export_sentences.txt',
                             n = 3000)
