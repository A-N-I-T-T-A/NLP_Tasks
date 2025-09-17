# Program 1: Naive Bayes with Bag of Words (BoW)

from sklearn.feature_extraction.text import CountVectorizer
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

# BoW vectorization
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Naive Bayes classification
model = MultinomialNB()
model.fit(X_train_bow, y_train)
y_pred = model.predict(X_test_bow)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot()
plt.title("Confusion Matrix - BoW")
plt.show()

from sklearn.metrics import classification_report, accuracy_score

# Evaluation metrics for BoW
print("=== Evaluation Report: BoW ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
