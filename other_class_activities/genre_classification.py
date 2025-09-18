import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
movies_df = pd.read_csv("movies.csv")
movies_df['primary_genre'] = movies_df['genres'].apply(lambda x: x.split('|')[0] if pd.notna(x) else 'Unknown')

# Features and labels
X = movies_df['title']
y = movies_df['primary_genre']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bag-of-Words Pipeline
bow_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
bow_pipeline.fit(X_train, y_train)
bow_preds = bow_pipeline.predict(X_test)
print("Bag-of-Words Accuracy:", accuracy_score(y_test, bow_preds))
print("Classification Report (BoW):\n", classification_report(y_test, bow_preds))

# TF-IDF Pipeline
tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
tfidf_pipeline.fit(X_train, y_train)
tfidf_preds = tfidf_pipeline.predict(X_test)
print("TF-IDF Accuracy:", accuracy_score(y_test, tfidf_preds))
print("Classification Report (TF-IDF):\n", classification_report(y_test, tfidf_preds))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Replace these with your actual accuracy values
bow_accuracy = accuracy_score(y_test, bow_preds)
tfidf_accuracy = accuracy_score(y_test, tfidf_preds)

# Prepare accuracy comparison data
accuracy_data = {
    'Model': ['Bag-of-Words', 'TF-IDF'],
    'Accuracy': [bow_accuracy, tfidf_accuracy]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Plot
plt.figure(figsize=(6, 4))
sns.barplot(data=accuracy_df, x='Model', y='Accuracy', palette='coolwarm')
plt.title('Accuracy Comparison of Bag-of-Words vs TF-IDF')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

# Get detailed reports as dict
bow_report = classification_report(y_test, bow_preds, output_dict=True)
tfidf_report = classification_report(y_test, tfidf_preds, output_dict=True)

top_genres = y.value_counts().nlargest(5).index.tolist()
genres = top_genres

# Gather F1 scores for both
bow_f1 = [bow_report[genre]["f1-score"] if genre in bow_report else 0 for genre in genres]
tfidf_f1 = [tfidf_report[genre]["f1-score"] if genre in tfidf_report else 0 for genre in genres]

# Create DataFrame for plotting
f1_df = pd.DataFrame({
    "Genre": genres,
    "Bag-of-Words": bow_f1,
    "TF-IDF": tfidf_f1
}).melt(id_vars="Genre", var_name="Model", value_name="F1-Score")

# Plot grouped bar chart
plt.figure(figsize=(8, 6))
sns.barplot(data=f1_df, x="Genre", y="F1-Score", hue="Model", palette="Set2")
plt.title("F1-Score Comparison (Top 5 Genres)")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
