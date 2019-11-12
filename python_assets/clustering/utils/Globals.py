import os
import argparse

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
ECBPLUS_DIR = os.path.join(ROOT_DIR, 'data', 'ecb_plus')
CLEAN_SENT_PATH = os.path.join(ROOT_DIR, 'data', 'ECBplus_coreference_sentences.csv')
parser = argparse.ArgumentParser(description='Process clustering args')
parser.add_argument('topics', type=str,
                    help='a comma delimited list of topics')
parser.add_argument('num_labels', type=int, nargs='?',
                    help='number of clusters for kmeans')
