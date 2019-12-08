import os
import json

ROOT = os.path.dirname(os.path.realpath(__file__))
EXTERNAL_VARS = os.path.join(ROOT, '..', '..', 'external_paths.json')
with open(EXTERNAL_VARS) as f:
    j = json.load(f)
    WORD_VECS_PATH = j['word_vecs']
