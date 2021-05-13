import sys
import fitz
import spacy
import numpy as np

nlp_model = spacy.load('nlp_model')
name = 'Alice Clark CV.pdf'


def showResult(fname):
    doc = fitz.open(fname)
    text = ""
    for page in doc:
        text = text + str(page.getText())

    tx = " ".join(text.split('\n'))
    # print(tx)

    doc = nlp_model(tx)

    for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}- {ent.text}')


showResult(name)
