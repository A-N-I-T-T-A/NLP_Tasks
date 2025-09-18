from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
doc1 = "Data science is fun"
doc2 = "Machine learning is fun"

# Create Bag-of-Words model
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform([doc1, doc2])

# Show BoW table
bow_table = vectors.toarray()
print("Bag-of-Words Table:")
print(vectorizer.get_feature_names_out())
print(bow_table)

# Calculate cosine similarity
cos_sim = cosine_similarity(vectors)
print("\nCosine Similarity Matrix:")
print(cos_sim)

# Interpretation
print(f"\nSimilarity between Doc1 and Doc2: {cos_sim[0][1]:.2f}")
