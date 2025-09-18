import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

text = "Alice visited Google headquarters in California on Monday."

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, "->", ent.label_)

displacy.serve(doc, style="ent")