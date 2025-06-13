import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For lemmatizer support

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sample text
text = """OpenAI plans to add Alphabet's Google cloud service to meet its growing needs for computing capacity, 
three sources told Reuters, marking a surprising collaboration between two prominent competitors in the artificial intelligence sector.
The deal, which has been under discussion for a few months, was finalised in May, one of the sources added.
It underscores how massive computing demands to train and deploy AI models are reshaping the competitive dynamics in AI, 
and marks OpenAI's latest move to diversify its compute sources beyond its major supporter Microsoft, including its high-profile 
Stargate data center project."""

# Tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

# Output
print("Sentences:", sentences)
print("Original Words:", words)
print("Filtered Words (No Stopwords):", filtered_words)
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)
