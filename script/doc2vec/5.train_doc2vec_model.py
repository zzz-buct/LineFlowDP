import os.path
import time
from script.my_util import *
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')

file_path = './corpus/'
model_path = './model/'

os.makedirs(file_path, exist_ok=True)


class DocumentDataset(object):
    def __init__(self, data: pd.DataFrame, column):
        document = data[column].apply(self.preprocess)
        self.documents = [
            TaggedDocument(text, [f"{row.release}::{row.filename}::{row.line_number}"])
            for _, (text, row) in enumerate(zip(document, data.itertuples()))
        ]

    def preprocess(self, document):
        # return preprocess_string(remove_stopwords(document))
        return document

    def __iter__(self):
        for document in self.documents:
            yield document

    def tagged_documents(self, shuffle=False):
        if shuffle:
            random.shuffle(self.documents)
        return self.documents


# read data
def readData(path, file):
    data = pd.read_csv(path + file, encoding='latin')
    return data


def train_doc2vec(project, method='lineflow'):
    if method == 'lineflow':
        data = readData(file_path, project + '.csv')
    elif method == 'noflow':
        data = readData(file_path, project + '_noflow.csv')
    elif method == 'linenoflow':
        data = readData(file_path, project + '_linenoflow.csv')
    data['code_line'] = data['code_line'].astype(str)
    document_dataset = DocumentDataset(data, 'code_line')
    docVecModel = Doc2Vec(min_count=1,
                          window=5,
                          vector_size=100,
                          sample=1e-4,
                          negative=5,
                          workers=2,
                          )
    docVecModel.build_vocab(document_dataset.tagged_documents())
    print('training......')
    docVecModel.train(document_dataset.tagged_documents(shuffle=False),
                      total_examples=docVecModel.corpus_count,
                      epochs=20)
    if method == 'lineflow':
        docVecModel.save(model_path + project + '_lineflow.d2v')
    elif method == 'linenoflow':
        docVecModel.save(model_path + project + '_linenoflow.d2v')
    elif method == 'noflow':
        docVecModel.save(model_path + project + '_noflow.d2v')
    print(f'{project} done!')


def main():
    total_time = 0
    count = 0
    for project in all_releases.keys():
        start_time = time.time()
        train_doc2vec(project=project, method='lineflow')
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        count += 1
        print(f"Project: {project}, Time: {elapsed_time} seconds")

    average_time = total_time / count
    print(f"Average Time: {average_time} seconds")


if __name__ == '__main__':
    main()
