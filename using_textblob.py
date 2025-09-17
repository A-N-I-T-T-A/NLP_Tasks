from textblob import TextBlob

text = "Hello Anitta! You're exploring NLP with TextBlob. It's pretty easy."

blob = TextBlob(text)

# Sentence tokenization
print(blob.sentences)

# Word tokenization
print(blob.words)

print("Tags:",blob.tags)
print("Nouns:",blob.noun_phrases)

for sentence in blob.sentences:
    print(sentence.sentiment)
    print(sentence.sentiment.polarity)