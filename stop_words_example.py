from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentance = "This is an example for showing how stop words work in nltk."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentance)

filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

## filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)