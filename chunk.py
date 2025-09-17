import nltk
from nltk import RegexpParser
from nltk.tree import Tree
from nltk.tokenize import TreebankWordTokenizer

# Download models (only first time)
nltk.download('averaged_perceptron_tagger')

# Initialize tokenizer
tokenizer = TreebankWordTokenizer()

text = "Alice in wonderland"

# Tokenize and POS tag
tokens = tokenizer.tokenize(text)
print(f"\n🔹 Tokens: {tokens}")

tags = nltk.pos_tag(tokens)
print(f"\n🔹 POS Tags: {tags}")

# Define noun phrase chunking pattern
grammar = r"""
    NP: {<NNP>+}
"""

# Parse chunks
chunker = RegexpParser(grammar)
tree = chunker.parse(tags)

# Show as plain-text tree
print("\n🔹 Chunked Tree (Text View):")
print(tree)

# Pretty print the tree in console
print("\n🔹 Pretty Tree:")
tree.pretty_print()

# OPTIONAL: Launch GUI tree window (works on local systems with GUI)
tree.draw()

