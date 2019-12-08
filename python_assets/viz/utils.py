import pandas as pd
import shared_vars as V
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# TODO: add beta test
def load_data():
    data = dict()

    f = 'model-doc_unbalanced-test.csv'
    index = pd.read_csv(os.path.join(V.DATA_DIR, f))
    data['md_ubt'] = merge_csv(index[index['experiment_id'].str.contains("5-cv")]['file_name'])

    f = 'model-doc_balanced-test.csv'
    index = pd.read_csv(os.path.join(V.DATA_DIR, f))
    data['md_bt'] = merge_csv(index[index['experiment_id'].str.contains("5-cv")]['file_name'])

    f = 'gold-doc_unbalanced-test.csv'
    index = pd.read_csv(os.path.join(V.DATA_DIR, f))
    data['gd_ubt'] = merge_csv(index[index['experiment_id'].str.contains("5-cv")]['file_name'])

    f = 'gold-doc_balanced-test.csv'
    index = pd.read_csv(os.path.join(V.DATA_DIR, f))
    data['gd_bt'] = merge_csv(index[index['experiment_id'].str.contains("5-cv")]['file_name'])
    return data


def merge_csv(fnames):
    df = pd.DataFrame()
    scale = ['bcub_precision', 'bcub_recall', 'bcub_f1',
              'ceafe_precision', 'ceafe_recall', 'ceafe_f1',
              'ceafm_precision', 'ceafm_recall', 'ceafm_f1',
              'muc_precision', 'muc_recall', 'muc_f1',
              'blanc_precision', 'blanc_recall', 'blanc_f1',
              'conll_f1']
    for f in fnames:
        part = pd.read_csv(os.path.join(V.DATA_DIR, f + '.csv'))
        part['true_percentage'] = part['this_true_pairs'] / (part['this_true_pairs'] + part['this_false_pairs'])
        part[scale] = part[scale]/100
        df = pd.concat([df, part], axis=0, ignore_index=True)

    return df

