from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize ## sentence tokenizer

sample = gutenberg.raw("bible-kjv.txt") ## raw() to get the text

tok = sent_tokenize(sample)

print(tok[5:15])

## C:\Users\Benjamin\AppData\Roaming\nltk_data\corpora