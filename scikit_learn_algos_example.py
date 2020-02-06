import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = []

for category in movie_reviews.categories(): # category is pos or neg
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) 

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words) # bool if in top 3000

    return features

#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = []
for (rev, category) in documents:
    featuresets.append((find_features(rev), category))

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# posterior = prior occurence x liklihood / evidence

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100) 
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100) 

# ERROR
#GNB_classifier = SklearnClassifier(GaussianNB())
#GNB_classifier.train(training_set)
#print("GNB_classifier accuracy percent:", (nltk.classify.accuracy(GNB_classifier, testing_set)) * 100) 

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set)) * 100) 

##LogisticRegression, SGDClassifier
##SVC, LinearSVC, NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)