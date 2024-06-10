### Thomas Middleton
from NewsGroupPreprocessing import GetNewsFileNames, RemoveNewsFormatting
import string 
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Preprocess the text
def TokeniseDoc(document):
    tokens = [word for word in document.lower().split() if word not in STOPWORDS]
    clean_tokens = []
    for token in tokens:
        clean_token = token.translate(punct_table).strip()
        if clean_token: clean_tokens.append(clean_token)
    return clean_tokens

def ProcessChunk(docs):
    tokenised_texts = [TokeniseDoc(doc) for doc in docs]
    dictionary.add_documents(tokenised_texts)
    return [dictionary.doc2bow(text) for text in tokenised_texts]

def NERForNewsGroup(NUMTOPICS):
    db_filenames = GetNewsFileNames()
    unformatted_text = RemoveNewsFormatting(db_filenames)
    corpus = ProcessChunk(unformatted_text)
    print("\n Data Imported \n")
    lda = LdaModel(corpus=corpus, num_topics=NUMTOPICS, id2word=dictionary, passes=15)
    print("\n LDA completed \n")
    return lda

punct_table = str.maketrans('', '', string.punctuation)
NUMTOPICS = 20
STOPWORDS = set(stopwords.words('english'))
#print(STOPWORDS)

dictionary = Dictionary()
lda = NERForNewsGroup(NUMTOPICS)

# Print the topics
for i, topic in lda.print_topics(num_topics=NUMTOPICS, num_words=3):
    print(f"Topic {i}: {topic}")
