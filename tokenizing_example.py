from nltk.tokenize import sent_tokenize, word_tokenize ## sentence and word tokenizer

example_text = "Hello my name is Benjamin, how are you? I am great, thanks for asking"

print(sent_tokenize(example_text))

print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)