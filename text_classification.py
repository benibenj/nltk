import nltk
import random
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories(): # category is pos or neg
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

#print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) # convert into a format that nlzk can use
print(all_words.most_common(15))

print(all_words["stupid"]) # how much times does this word get used in the reviews