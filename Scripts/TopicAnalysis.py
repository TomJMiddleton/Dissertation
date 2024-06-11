### Thomas Middleton
from NewsGroupPreprocessing import GetNewsFileNames, RemoveNewsFormatting
import string 
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Preprocess the text
def TokeniseDoc(document):
    tokens = [word for word in document.lower().split()]
    clean_tokens = []
    for token in tokens:
        clean_token = token.translate(punct_table).strip()
        if clean_token and clean_token not in STOPWORDS:
            clean_tokens.append(clean_token)
    return clean_tokens

def ProcessChunk(docs):
    tokenised_texts = [TokeniseDoc(doc) for doc in docs]
    dictionary.add_documents(tokenised_texts)
    return [dictionary.doc2bow(text) for text in tokenised_texts]

def ExportLDAModel(path, name):
    # Save the model, dictionary, corpus
    lda.save(path + name + '_lda_model.gensim')
    dictionary.save(path + name + '_lda_dictionary.gensim')
    MmCorpus.serialize(path + name + 'lda_corpus.mm', corpus)

def ImportLDAModel(path, name):
    # Load the model, dictionary, corpus
    lda_loaded = LdaModel.load(path + name + '_lda_model.gensim')
    dictionary_loaded = Dictionary.load(path + name + '_lda_dictionary.gensim')
    corpus_loaded = MmCorpus(path + name + '_lda_corpus.mm')
    return lda_loaded, dictionary_loaded, corpus_loaded

def NERForNewsGroup(NUMTOPICS):
    db_filenames = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_filenames)
    corpus = ProcessChunk(unformatted_text)
    print("\n Data Imported \n")
    lda = LdaModel(corpus=corpus, num_topics=NUMTOPICS, id2word=dictionary, passes=15)
    print("\n LDA completed \n")
    return lda, corpus

punct_table = str.maketrans('', '', string.punctuation)
SAVEPATH = "Datasets/Processed/20NG/"
NUMTOPICS = 20
STOPWORDS = set(stopwords.words('english'))
for i in range(10):
    STOPWORDS.add(str(i))
#print(STOPWORDS)

dictionary = Dictionary()
lda, corpus = NERForNewsGroup(NUMTOPICS)

# Print the topics
for i, topic in lda.print_topics(num_topics=NUMTOPICS, num_words=3):
    print(f"Topic {i}: {topic}")


ExportLDAModel(SAVEPATH, "TestOne")