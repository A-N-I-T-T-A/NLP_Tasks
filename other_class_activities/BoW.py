from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample documents
docs = [
    "Data science is fun and exciting",
    "Machine learning is a branch of data science",
    "Natural language processing is a field of AI",
    "Deep learning is a subfield of machine learning"
]

# Vectorize using CountVectorizer (BoW)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

# Plot word clouds for each document
num_docs = len(docs)
cols = 2
rows = (num_docs + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))

for i in range(num_docs):
    ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
    bow_vector = X[i].toarray().flatten()
    word_freq = dict(zip(feature_names, bow_vector))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_freq)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f"BoW Word Cloud - Doc{i+1}")

# Hide any unused subplots
for j in range(num_docs, rows * cols):
    fig.delaxes(axes[j // cols, j % cols] if rows > 1 else axes[j % cols])

plt.tight_layout()
plt.show()
