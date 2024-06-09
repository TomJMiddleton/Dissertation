### Thomas Middleton

from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Example corpus
documents = [
    "Basketball is a popular sport.",
    "Soccer players can score goals.",
    "Tennis is played with a racket.",
    "Basketball players score points by shooting the ball through the hoop.",
    "Soccer is known as football in many countries."
]

# Preprocess the text
texts = [[word for word in document.lower().split()] for document in documents]

# Create a dictionary representation of the documents
dictionary = Dictionary(texts)

# Convert dictionary to a bag of words corpus
corpus = [dictionary.doc2bow(text) for text in texts]

# Train the LDA model
lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Print the topics
for i, topic in lda.print_topics(num_topics=3, num_words=3):
    print(f"Topic {i}: {topic}")