import spacy
import pickle
import random

train_datas = pickle.load(open('train_data.pkl', 'rb'))

nlp = spacy.blank('en')


def train_model():
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)


        #ner = nlp.add_pipe('ner', last=True)

        return ner
    else:
        ner = nlp.get_pipe('ner')
        return ner


def nerData(ner, train_data):
    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    for pip in nlp.pipe_names:
        print(pip)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Statring iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    pass

            print(losses)


ners = train_model()

nerData(ners, train_datas)

nlp.to_disk('nlp_model')
