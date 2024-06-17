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
    document = document.translate(punct_table).strip()
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


# Token filter
NBELOW = 12
NOVER = 0.3
NKEEP = 60000

NUMTOPICS = 20
STOPWORDS = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)

SAVEPATH = "Datasets/Processed/20NG/"

for i in range(10):
    STOPWORDS.add(str(i))
#print(STOPWORDS)

dictionary = Dictionary()
lda, corpus = NERForNewsGroup(NUMTOPICS)

# Print the topics
for i, topic in lda.print_topics(num_topics=NUMTOPICS, num_words=3):
    print(f"Topic {i}: {topic}")


ExportLDAModel(SAVEPATH, "TestOne")