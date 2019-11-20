import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

def read_sample(fn="../../data/test_sample.xlsx"):
    data = pd.read_excel(fn)
    return data

def encode_label(multilabels):
    encoded_labels = []
    for multilabel in multilabels:
        encoded_labels.append([label.strip("'| ") for label in multilabel[1:-1].split(",")])

    binarizer = MultiLabelBinarizer()
    encoded_labels = binarizer.fit_transform(encoded_labels)
    return encoded_labels, binarizer.classes_