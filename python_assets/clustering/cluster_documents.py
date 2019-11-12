from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation
from collections import defaultdict
import re
from utils import Globals as glob
from utils.ECBWrapper import ECBWrapper
import sys


args = glob.parser.parse_args()
topics = args.topics.split(' ')
num_labels = args.num_labels

ecb_wrapper = ECBWrapper(glob.ECBPLUS_DIR, topics=None, lemmatize=False)
docs, targets, f_names = ecb_wrapper.make_data_for_clustering(option='text',
                                                     topics=topics,
                                                     sub_topics=True, filter=True)

vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1,3), stop_words='english',max_features=500)
X = vectorizer.fit_transform(docs)

if num_labels:
    clustering_obj = KMeans(n_clusters=num_labels, init='k-means++', max_iter=200,
                            n_init=20, random_state=665,
                            n_jobs=20, algorithm='auto').fit(X)
else:
    clustering_obj = AffinityPropagation(damping=0.5).fit(X)
clusters = defaultdict(list)
for k,v in zip(clustering_obj.labels_, f_names):
    clusters[k].append(v)
clusters = {k:' '.join(v) for k,v in clusters.items()}
hcv = metrics.homogeneity_completeness_v_measure(targets, clustering_obj.labels_)
ari = metrics.adjusted_rand_score(targets, clustering_obj.labels_)
res = {'homogeneity': round(hcv[0], 4),
       'completeness': round(hcv[1],4),
       'v-measure': round(hcv[2],4),
       'ari': round(ari,4)}
pattern = re.compile('{|}')
sys.stdout.write(pattern.sub('', str(clusters)))
sys.stdout.write('BREAK')
sys.stdout.write(pattern.sub('', str(res)))