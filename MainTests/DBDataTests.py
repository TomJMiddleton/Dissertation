import sqlite3

def ReviewSentenceTableInTxtFile(db_path, output_file, n):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Execute the query to get the first n sentences
    cur.execute("SELECT sent FROM Sentence LIMIT ?", (n,))
    
    # Fetch all results
    sentences = cur.fetchall()
    
    # Write sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence[0] + '\n')
    
    # Close the database connection
    conn.close()
    
    print(f"{len(sentences)} sentences have been written to {output_file}")



ReviewSentenceTableInTxtFile(db_path='./Datasets/Database/NewsGroupDB.db',
                             output_file = 'db_export_sentences.txt',
                             n = 3000)
