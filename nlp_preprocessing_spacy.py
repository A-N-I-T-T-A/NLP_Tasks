import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """OpenAI plans to add Alphabet's Google cloud service to meet its growing needs for computing capacity, 
three sources told Reuters, marking a surprising collaboration between two prominent competitors in the artificial intelligence sector.
The deal, which has been under discussion for a few months, was finalised in May, one of the sources added.
It underscores how massive computing demands to train and deploy AI models are reshaping the competitive dynamics in AI, 
and marks OpenAI's latest move to diversify its compute sources beyond its major supporter Microsoft, including its high-profile 
Stargate data center project."""

# Process the text
doc = nlp(text)

# Sentence Tokenization
print("Sentences:",[sent for sent in doc.sents])


# Word Tokenization
print("Tokens:", [token.text for token in doc])


# Stop Word Removal
filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]

# Lemmatization (after stop word removal)
lemmatized_tokens = [token.lemma_ for token in filtered_tokens]

# Final Output
print("\nFiltered Tokens (No Stopwords):", [token.text for token in filtered_tokens])
print("Lemmatized Tokens:", lemmatized_tokens)
