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

from sklearn.feature_extraction.text import CountVectorizer

# Split data
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.3, random_state=42)

# Vectorize with BoW
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Train SVM classifier
svm_bow = SVC(kernel='linear')
svm_bow.fit(X_train_bow, y_train)
y_pred_bow = svm_bow.predict(X_test_bow)

# Confusion matrix
cm_bow = confusion_matrix(y_test, y_pred_bow)
disp_bow = ConfusionMatrixDisplay(confusion_matrix=cm_bow, display_labels=["Negative", "Positive"])
disp_bow.plot()
plt.title("Confusion Matrix - SVM with BoW")
plt.show()


from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred_bow ))
print(classification_report(y_test, y_pred_bow, target_names=["Negative", "Positive"]))
