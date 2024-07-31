import sqlite3, os, re
from contextlib import closing
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize
from CreateLocalDB import InstatiateDB
from NewsGroupPreprocessing import ExtractDocumentIndex, ExtractDocumentText

document_re = re.compile(r'^Newsgroup:.*?(?=^Newsgroup:|\Z)', re.DOTALL | re.MULTILINE)
NEWSGROUPDBFILEPATH = './Datasets/Database/NewsGroupDB.db'

def NewsGroupDocumentExtraction():
    dir_names = os.listdir("./Datasets/Raw/20NG/")
    assert 'alt.atheism.txt' in dir_names, " NewsGroup Filepath is incorrect"

    news_filenames =  [file for file in dir_names if file.endswith('.txt')]
    newsgroup_docs = []
    
    for filename in news_filenames:
        with open(os.path.join("./Datasets/Raw/20NG/", filename), 'r') as file:
            content = file.read()
            documents = document_re.findall(content)
            for doc in documents:
                newsgroup, document_id = ExtractDocumentIndex(doc)
                if newsgroup and document_id:
                    doc_text = ExtractDocumentText(doc)
                    doc_title = newsgroup + document_id
                    newsgroup_docs.append([doc_title, doc_text])

    newsgroup_doc_df = pd.DataFrame(newsgroup_docs, columns=['title', 'document'])
    return newsgroup_doc_df


def NewsGroupSentenceExtraction(newsgroup_doc_df):
    newsgroup_doc_df['sent'] = newsgroup_doc_df['document'].apply(sent_tokenize)
    sentences_df = newsgroup_doc_df.reset_index().rename(columns={'index': 'DocID'})
    sentences_df['DocID'] += 1
    sentences_df = sentences_df.explode('sent')
    sentences_df = sentences_df[['DocID', 'sent']]
    print(sentences_df.head(200))
    return sentences_df


def WriteDataframeToDatabase(df_to_write, table_name, db_filepath):
    with closing(sqlite3.connect(db_filepath)) as conn:
        df_to_write.to_sql(table_name, conn, if_exists='append', index=False)

if __name__ == "__main__":
    # punkt sentence tokenizer
    nltk.download('punkt')

    # Ensure the database exists
    InstatiateDB(NEWSGROUPDBFILEPATH)

    # Populate the Document table in the database
    newsgroup_doc_df = NewsGroupDocumentExtraction()
    #WriteDataframeToDatabase(newsgroup_doc_df, "Document", NEWSGROUPDBFILEPATH)

    # Populate the Sentence table in the database
    sentences_df = NewsGroupSentenceExtraction(newsgroup_doc_df)
    #WriteDataframeToDatabase(sentences_df, "Sentence", NEWSGROUPDBFILEPATH)

    print(f"\nTotal number of sentences: {len(sentences_df)}")
    print(f"Number of unique documents: {sentences_df['DocID'].nunique()}")

    # Populate the Keyword table in the database