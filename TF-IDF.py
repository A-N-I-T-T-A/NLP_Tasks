from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Define documents
docs =  [
    "Data science is fun and exciting",
    "Machine learning is a branch of data science",
    "Natural language processing is a field of AI",
    "Deep learning is a subfield of machine learning"
]

# Step 2: Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

# Step 3: Cosine similarity
similarity_matrix = cosine_similarity(X)
doc_labels = [f"Doc{i+1}" for i in range(len(docs))]
df_sim = pd.DataFrame(similarity_matrix, index=doc_labels, columns=doc_labels)

# Step 4: Heatmap of similarities
plt.figure(figsize=(6, 5))
sns.heatmap(df_sim, annot=True, cmap="Blues")
plt.title("TF-IDF Document Similarity")
plt.show()

# Step 5: Word clouds for each document
num_docs = len(docs)
cols = 3
rows = (num_docs + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

for i in range(num_docs):
    ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
    tfidf_vector = X[i].toarray().flatten()
    word_scores = dict(zip(feature_names, tfidf_vector))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_scores)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f"TF-IDF Word Cloud - Doc{i+1}")

# Hide empty subplots
for j in range(num_docs, rows * cols):
    fig.delaxes(axes[j // cols, j % cols] if rows > 1 else axes[j % cols])

plt.tight_layout()
plt.show()
