from os import walk
import os
from os.path import isfile, join
from ECBDocWrapper import ECBDocWrapper
import shared_vars as vars
import sys


total_files = 0
for root, subdirs, files in walk(vars.ECB_PATH):
    for f in files:
        if isfile(join(root, f)) and f.endswith('.xml'):
            total_files += 1
print(total_files, 'total files')
done = 0
for root, subdirs, files in walk(vars.ECB_PATH):
    for f in files:
        if isfile(join(root, f)) and f.endswith('.xml'):
            ecb_doc = ECBDocWrapper(join(root, f))
            ecb_doc.augment_ecb_tokens()
            done += 1
            print(str(done) + '/' + str(total_files))
