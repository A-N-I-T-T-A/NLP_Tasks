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

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

# Split data
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3, random_state=42)

# Vectorize with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVM classifier
svm_tfidf = SVC(kernel='linear')
svm_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)

# Confusion matrix
cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
disp_tfidf = ConfusionMatrixDisplay(confusion_matrix=cm_tfidf, display_labels=["Negative", "Positive"])
disp_tfidf.plot()
plt.title("Confusion Matrix - SVM with TF-IDF")
plt.show()


from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf, target_names=["Negative", "Positive"]))
