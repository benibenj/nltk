from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

for w in example_words:
    print(ps.stem(w))

new_text = "It is very important to enjoy every day, because enyoing it makes you happy and you will remember the enjoyed days for long."

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))