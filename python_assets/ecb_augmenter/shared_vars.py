import os

ROOT = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(ROOT, '..', '..', 'data')
ECB_PATH = os.path.join(ROOT, 'ECB+')
CLEAN_SENT_PATH = os.path.join(DATA_DIR, 'ECBplus_coreference_sentences.csv')
JAVA_ASSETS = os.path.join(ROOT, '..', '..', 'java_assets')
CAEVO_PATH = os.path.join(JAVA_ASSETS, 'caevo')
TXT_PATH = os.path.join(ROOT, 'sentence.txt')
CAEVO_ARGS = ['mvn', 'exec:java', '-Dexec.mainClass=caevo.Main',
              '-Dprops=default.properties', '-Dsieves=default.sieves',
              '-Dexec.args=\"\" {0} raw\"\"'.format(TXT_PATH)]
CAEVO_OUTPUT = os.path.join(ROOT, 'sentence.txt.info.xml')

ECB_AUG_DIR = os.path.join(DATA_DIR, 'ecb_aug')
if not os.path.exists(ECB_AUG_DIR):
    os.mkdir(ECB_AUG_DIR)
