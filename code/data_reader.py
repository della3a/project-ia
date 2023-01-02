# Read data in csv format

import pandas as pd


def read_data(path):
    data = pd.read_csv(path, sep=',', encoding='utf-8')
    return data
