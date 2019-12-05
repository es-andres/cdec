from os import walk
import os
from os.path import isfile, join
from ECBDocWrapper import ECBDocWrapper
import shared_vars as vars
import sys
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_labels(ax, data):

    # # The x position of the median line
    # xpos = med.get_xdata()
    #
    # # Lets make the text have a horizontal offset which is some
    # # fraction of the width of the box
    # xoff = 0.10 * (xpos[1] - xpos[0])

    # The x position of the labels
    # xlabel = xpos[1] + xoff

    # The median is the y-position of the median line
    median = np.quantile(data,  0.5)
    # The 25th and 75th percentiles are found from the
    # top and bottom (max and min) of the box
    pc25 = np.quantile(data,  0.25)
    pc75 = np.quantile(data,  0.75)

    # Make some labels on the figure using the values derived above
    ax.text(500, .6,
            '{:6.3g}'.format(median), va='center')
    # ax.text(pc25, .6,
    #         '{:6.3g}'.format(pc25), va='center')
    # ax.text(pc75, .6,
    #         '{:6.3g}'.format(pc75), va='center')


def augment_files():
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


def ev_pred_perf():
    global_res = {'tp': 0,
                  'fp': 0,
                  'tn': 0,
                  'fn': 0}
    for root, subdirs, files in walk(vars.ECB_AUG_DIR):
        for f in files:
            if isfile(join(root, f)) and f.endswith('.xml'):
                ecb_doc = ECBDocWrapper(join(root, f))
                local_res = ecb_doc.calculate_ev_pred_performance()
                global_res = {k: global_res[k] + local_res[k] for k in local_res.keys()}
    precision = global_res['tp'] / (global_res['tp'] + global_res['fp'])
    recall = global_res['tp'] / (global_res['tp'] + global_res['fn'])
    f1 = 2*((precision*recall) / (precision + recall))
    accuracy = (global_res['tp'] + global_res['tn']) / sum([global_res[k] for k in global_res.keys()])
    print('precision: {0}\nrecall: {1}\nf1: {2}\naccuracy: {3}'.format(precision, recall, f1, accuracy))


def evs_per_doc():
    num_evs = []
    for root, subdirs, files in walk(vars.ECB_AUG_DIR):
        for f in files:
            if isfile(join(root, f)) and f.endswith('.xml'):
                ecb_doc = ECBDocWrapper(join(root, f))
                evs, local_chains = ecb_doc.count_evs_and_chains()
                num_evs.append(evs)
    make_boxplot(num_evs, '# Events in Document', 'evs.png')


def count_evs_and_chains():
    num_evs = 0
    chains = dict()
    for root, subdirs, files in walk(vars.ECB_AUG_DIR):
        for f in files:
            if isfile(join(root, f)) and f.endswith('.xml'):
                ecb_doc = ECBDocWrapper(join(root, f))
                evs, local_chains = ecb_doc.count_evs_and_chains()
                num_evs += evs
                for k in local_chains.keys():
                    if k in chains:
                        chains[k] = chains[k] + local_chains[k]
                    else:
                        chains[k] = local_chains[k]
    print('num_evs:', num_evs)
    chain_lengths = [len(chains[k]) for k in chains.keys()]
    print('chain_stats:', stats.describe(chain_lengths))
    print('q25:', np.quantile(chain_lengths, 0.25))
    print('q50:', np.quantile(chain_lengths, 0.50))
    print('q75:', np.quantile(chain_lengths, 0.75))
    make_boxplot(chain_lengths, 'Length of Coreference Chain', 'chains.png')


def docs_per_clust():
    docs = dict()
    for root, subdirs, files in walk(vars.ECB_AUG_DIR):
        for f in files:
            if isfile(join(root, f)) and f.endswith('.xml'):
                name = f.split('_')
                topic = name[0] + '_' + ''.join([i for i in name[1] if not i.isdigit()])
                if topic not in docs:
                    docs[topic] = 0
                docs[topic] += 1
    top_docs = [docs[k] for k in docs.keys()]
    print(stats.describe(top_docs))
    print('std dev',  np.sqrt(stats.describe(top_docs).variance))
    print('q25:', np.quantile(top_docs, 0.25))
    print('q50:', np.quantile(top_docs, 0.50))
    print('q75:', np.quantile(top_docs, 0.75))
    make_boxplot(top_docs, '# Documents in Sub-Topic', 'docs.png')


def words_per_doc():
    words = []
    for root, subdirs, files in walk(vars.ECB_AUG_DIR):
        for f in files:
            if isfile(join(root, f)) and f.endswith('.xml'):
                ecb_doc = ECBDocWrapper(join(root, f))
                words.append(len(ecb_doc.get_all_tokens()))
    print('num_worods', sum(words))
    print(stats.describe(words))
    print('std dev',  np.sqrt(stats.describe(words).variance))
    print('q25:', np.quantile(words, 0.25))
    print('q50:', np.quantile(words, 0.50))
    print('q75:', np.quantile(words, 0.75))
    make_boxplot(words, '# Words in Document', 'words.png')


def make_boxplot(data, label, fname):
    fig = plt.figure(figsize=(3, 5), dpi=500)
    ax = fig.add_subplot()
    plt.boxplot(data, False, vert=True, widths=0.75, showmeans=True)
    # major_ticks = np.arange(0, np.max(words), 5)
    # ax.set_xticks(major_ticks)
    ax.set_xticks([])
    plt.xlabel(label, fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, transparent=True)





# augment_files() # to augment ecb+ files
# ev_pred_perf() # calculate event detection performance
evs_per_doc()
count_evs_and_chains() # chain and ev stats
docs_per_clust()
words_per_doc()