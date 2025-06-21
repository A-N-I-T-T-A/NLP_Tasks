from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

docs =  [
    "Data science is fun and exciting",
    "Machine learning is a branch of data science",
    "Natural language processing is a field of AI",
    "Deep learning is a subfield of machine learning"
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Convert to frequency dictionary
word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

# Create WordCloud
wordcloud = WordCloud().generate_from_frequencies(word_freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
