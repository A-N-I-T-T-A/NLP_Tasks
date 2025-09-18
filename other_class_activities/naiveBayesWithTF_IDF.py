# Program 2: Naive Bayes with TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Sample data
sentences = [
    'I love this movie',
    'This movie is terrible',
    'I really enjoyed this film',
    'This film is awful',
    'What a fantastic experience',
    'I hated this film',
    'This was a great movie',
    'The film was not good',
    'I am very happy with this movie',
    'I am disappointed with this film'
]

# Labels: 1 = Positive, 0 = Negative
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes classification
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot()
plt.title("Confusion Matrix - TF-IDF")
plt.show()

from sklearn.metrics import classification_report, accuracy_score

# Evaluation metrics for TF-IDF
print("=== Evaluation Report: TF-IDF ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
