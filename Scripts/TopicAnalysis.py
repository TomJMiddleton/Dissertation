### Thomas Middleton
from NewsGroupPreprocessing import GetNewsFileNames, RemoveNewsFormatting
import string 
import csv
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Preprocess the text
def TokeniseDoc(document):
    document = document.translate(PUNCT_TABLE).strip()
    tokens = [word for word in document.lower().split()]
    clean_tokens = []
    for token in tokens:
        if len(token) > 1 and token not in STOPWORDS and token.isdigit() == False:
            clean_tokens.append(token)
    return clean_tokens

def ProcessChunk(docs):
    tokenised_texts = [TokeniseDoc(doc) for doc in docs]
    dictionary.add_documents(tokenised_texts)
    dictionary.filter_extremes(no_below=NBELOW, no_above=NOVER, keep_n=NKEEP)
    return [dictionary.doc2bow(text) for text in tokenised_texts]

def ExportLDAModel(path, name):
    # Save the model, dictionary, corpus
    lda.save(path + name + '_lda_model.gensim')
    dictionary.save(path + name + '_lda_dictionary.gensim')
    MmCorpus.serialize(path + name + '_lda_corpus.mm', corpus)

def ImportLDAModel(path, name):
    # Load the model, dictionary, corpus
    lda_loaded = LdaModel.load(path + name + '_lda_model.gensim')
    dictionary_loaded = Dictionary.load(path + name + '_lda_dictionary.gensim')
    corpus_loaded = MmCorpus(path + name + '_lda_corpus.mm')
    return lda_loaded, dictionary_loaded, corpus_loaded

def NERForNewsGroup(NUMTOPICS):
    db_save_file_names = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_save_file_names)
    corpus = ProcessChunk(unformatted_text)
    print("\n Data Imported \n")
    lda = LdaModel(corpus=corpus, num_topics=NUMTOPICS, id2word=dictionary, passes=15)
    print("\n LDA completed \n")
    return lda, corpus

def PrintTopics(lda, n_words):
    for i, topic in lda.print_topics(num_topics=NUMTOPICS, num_words=n_words):
        print(f"Topic {i}: {topic}")

def GetDocumentTopics(document, n_topics = 2):
    tokens = TokeniseDoc(document)
    bag_of_words = dictionary.doc2bow(tokens)
    topic_distribution = lda.get_document_topics(bag_of_words)
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    sorted_topics = sorted_topics[:n_topics]
    return sorted_topics

def WriteNGTopics(n_topics):
    db_save_file_names = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_save_file_names)
    save_file_path = "Datasets/Processed/20NG/"
    print("Documents Imported")
    with open(save_file_path + 'test_topics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for doc in unformatted_text:
            topics = GetDocumentTopics(doc, )
            doc_topics = [ITEM for pair in topics for ITEM in pair]
            writer.writerow(doc_topics, n_topics)
    print(f" Written {n_topics} topics per document to csv")



# CONSTANTS -----------------------------------------------------------------------------
NBELOW = 12
NOVER = 0.3
NKEEP = 60000

NTOPICSTOWRITE = 2

NUMTOPICS = 20
STOPWORDS = set(stopwords.words('english'))
PUNCT_TABLE = str.maketrans('', '', string.punctuation)

# Variables -----------------------------------------------------------------------------
dictionary = Dictionary()
save_file_path = "Datasets/Processed/20NG/"
save_file_name = "TestOne"

# Code ----------------------------------------------------------------------------------

#lda, corpus = NERForNewsGroup(NUMTOPICS)

#ExportLDAModel(save_file_path, save_file_name)

lda, dictionary, corpus = ImportLDAModel(save_file_path, save_file_name)

WriteNGTopics(NTOPICSTOWRITE)