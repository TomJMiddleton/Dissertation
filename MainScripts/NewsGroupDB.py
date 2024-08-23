import sqlite3, os, re
from contextlib import closing
import pandas as pd
import nltk
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from CreateLocalDB import InstatiateDB
from NewsGroupPreprocessing import ExtractDocumentIndex, ExtractDocumentText, CleanNewsGroupBasics

document_re = re.compile(r'^Newsgroup:.*?(?=^Newsgroup:|\Z)', re.DOTALL | re.MULTILINE)
NEWSGROUPDBFILEPATH = './Datasets/FinalDB/FinalSQLDB.db'

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
                    #with open('doc_txt.txt', 'w', encoding='utf-8') as f:
                    #    f.write(doc_text)
                    doc_text_clean = CleanNewsGroupBasics(doc_text)
                    #with open('doc_text_clean.txt', 'w', encoding='utf-8') as f:
                    #    f.write(doc_text_clean)
                    doc_title = newsgroup + document_id
                    newsgroup_docs.append([doc_title, doc_text_clean])

    newsgroup_doc_df = pd.DataFrame(newsgroup_docs, columns=['Title', 'CleanedDocument'])
    return newsgroup_doc_df


def NewsGroupSentenceExtraction(newsgroup_doc_df):
    tokenizer = AutoTokenizer.from_pretrained('huggingface-course/bert-finetuned-ner')
    # Tokenize sentences from the cleaned document
    newsgroup_doc_df['SentenceText'] = newsgroup_doc_df['CleanedDocument'].apply(sent_tokenize)
    
    # Create a new DataFrame to store sentences with DocID
    sentences_df = newsgroup_doc_df.reset_index().rename(columns={'index': 'DocID'})
    sentences_df['DocID'] += 1
    sentences_df = sentences_df.explode('SentenceText')
    sentences_df = sentences_df[['DocID', 'SentenceText']]
    
    # Remove sentences with values: None, whitespace, only punctuation
    def CheckNullSentences(text):
        stripped_text = str(text).strip()
        return bool(re.match(r'^[\W\s]*$', stripped_text))
    
    sentences_df = sentences_df.dropna(subset=['SentenceText'])
    sentences_df = sentences_df[~sentences_df['SentenceText'].apply(CheckNullSentences)]
    sentences_df = sentences_df.reset_index(drop=True)
    
    # Chunk sentences together with a limit of <500 tokens per chunk
    chunks = []
    current_chunk = []
    current_token_count = 0
    current_doc_id = None

    for _, row in sentences_df.iterrows():
        sentence = row['SentenceText']
        doc_id = row['DocID']
        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
        
        if current_token_count + token_count < 500 and (current_doc_id is None or current_doc_id == doc_id):
            current_chunk.append(sentence)
            current_token_count += token_count
            current_doc_id = doc_id
        else:
            chunks.append({'DocID': current_doc_id, 'SentenceText': ' '.join(current_chunk)})
            current_chunk = [sentence]
            current_token_count = token_count
            current_doc_id = doc_id

    # Add the last chunk if there are remaining sentences
    if current_chunk:
        chunks.append({'DocID': current_doc_id, 'SentenceText': ' '.join(current_chunk)})
    
    # Create a new DataFrame for the chunks
    chunks_df = pd.DataFrame(chunks)
    
    return chunks_df


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
    #print(newsgroup_doc_df.head(10))
    WriteDataframeToDatabase(newsgroup_doc_df, "Documents", NEWSGROUPDBFILEPATH)

    # Populate the Sentence table in the database
    sentences_df = NewsGroupSentenceExtraction(newsgroup_doc_df)
    WriteDataframeToDatabase(sentences_df, "Sentences", NEWSGROUPDBFILEPATH)

    print(f"\nTotal number of sentences: {len(sentences_df)}")
    print(f"Number of unique documents: {sentences_df['DocID'].nunique()}")

    # Populate the Keyword table in the database