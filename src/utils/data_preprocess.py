import pandas as pd

def read_sample(fn="../../data/test_sample.xlsx"):
    data = pd.read_excel(fn)
    return data