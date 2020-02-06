from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better")) # default is noun (n)
print(lemmatizer.lemmatize("better", pos="a")) # adjective
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", pos="v")) # verb

# simmular to stemming but probably better