from os.path import isfile, join
import os
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import wordnet,stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import sys
from . import Globals
import spacy
from sklearn.model_selection import KFold


class ECBWrapper:
    """
    Wrapper for the ECB+ corpus.

        input:
        - dir_ : path to root directory of the corpus
        - topics: a list of str that can subset the corpus

        class_vars:
        - all_files: flat list of all the files in the current subset
    """

    def __init__(self, dir_,lemmatize,topics=None):
        self.root_dir = dir_
        self.topics = topics
        # for root, subdirs, files in walk(dir_)
        self.all_files = []
        self.lemmatize = lemmatize
        self.nlp = spacy.load('en')

        for root, subdirs, files in os.walk(dir_):
            for f in files:
                if isfile(join(root,f)) and f.endswith('naf'):
                    if self.topics is not None:
                        if self.get_topic_num(join(root,f))[0] in self.topics:
                            self.all_files.append(join(root, f))
                    else:
                        self.all_files.append(join(root, f))
        self.annotated = dict()
        with open(Globals.CLEAN_SENT_PATH) as f:
            for line in f.readlines()[1:]:
                line = line.split(',')
                f_name = line[0] + '_' + line[1]
                if f_name not in self.annotated:
                    self.annotated[f_name] = []
                self.annotated[f_name].append(line[2].replace('\n',''))

    def make_data_for_clustering(self, option, topics, filter=False,sub_topics=False):
        documents = []
        targets = []
        f_names = []
        # get all files in all topics, partitioned by suptopic
        for i in topics:
            if i not in [15, 17]:
                # get all files in topic i
                for f in self.get_topic(topic=i):
                    if sub_topics:
                        label = self.get_topic_num(f)
                        label = label[0] + "_" + label[1]
                    else:
                        label = str(i)
                    documents.append(self.get_text(f, element_type=option, filter=filter))
                    targets.append(label)
                    f_names.append(os.path.basename(f))

        return documents, targets, f_names

    '''
    Gives all paths in a given topic, sub-topic. 
    If sup_topic is not passed, all files in the topic are returned
    
        input:
        - topic: str of desired topic
        - sub_topic: str of desired sub topic
        
        output: 
        - flat list of files
    '''
    def get_topic(self, topic, sub_topic=-1):
        files = []
        if sub_topic == -1:
            for f in self.all_files:
                if os.path.basename(f).split("_")[0] == str(topic):
                    files.append(f)
        else:
            for f in self.all_files:
                # ecb
                if sub_topic == 1:
                    if os.path.basename(f).split("_")[0] == str(topic) and not 'plus' in f:
                        files.append(f)
                # ecb+
                elif sub_topic == 2:
                     if os.path.basename(f).split("_")[0] == str(topic) and 'plus' in f:
                         files.append(f)
        return files

    '''
    Gives files for a subset of topics (includes sub-topic in a dictionary)
    
        input: 
        - topics: list to subset topics (if None it gives all topics)
        
        output: 
        - dictionary of dictionaries, where the first level key = topic, first level value = dict of 
        subtopics, with key = sub_topic, value = list of paths 
    '''
    def get_files_by_topic(self, topics= None):
        files_by_topic = dict()
        topic_nums = range(1,46)
        if topics is not None:
            topic_nums = topics
        # get all files in all topics, partitioned by suptopic
        for i in topic_nums:
            if i not in [15, 17]:
                files_by_topic[str(i)] = {'1': [], '2': []}
                # get all files in topic i
                topic_files = self.get_topic(topic=i)
                for f in topic_files:
                    top = self.get_topic_num(f)
                    files_by_topic[top[0]][top[1]].append(f)
        return files_by_topic

    '''
    Gives topic,subtopic for a given path 
    
        input: 
        - path: path you want topic number from 
        
        output: 
        - tuple, where tuple[0] = topic, tuple[1] = subtopic
    
    '''
    def get_topic_num(self, path):
        p = os.path.basename(path).split("_")
        top = p[0]
        sub = 'ecbplus' if 'plus' in p[1] else 'ecb'
        return top, sub


    '''
    Return different types of text representations of an ECB+ document 
    
        input: 
        - path: path of desired document
        - lemmatize: boolean asking for lemmatized/non-lemmatized text 
        - element_type: None gives full text, 
            * 'event_trigger' gives event triggers
            * 'doc_template' gives all ecb annotations (event triggers, human/non-human participans, times, locations)
            * 'events_and_participants' gives events and human/non-human participants
            * 'events_participants_locations' gives events, human/non-human participants and locations
            * 'participants' gives human/non-human participants
        
        output: 
        - a string (if element_type=None) or list of desired types of elements
    '''

    def filter_doc(self, doc):

        # parsed = self.nlp(' '.join(doc))

        parsed = self.nlp.tokenizer.tokens_from_list(doc)
        filtered_tokens = [tok.text for tok in parsed if tok.ent_iob_ != 'O']
        if len(filtered_tokens) < 3:
            filtered_doc = doc
        else:
            filtered_doc = ' '.join(filtered_tokens)
        return filtered_doc

    def get_text(self, path, filter=False, element_type=None):
        def check(tag, tags):
            for t in tags:
                if tag.startswith(t):
                    return True
            return False
        pattern = re.compile('[\W_]+', re.UNICODE)
        # text
        if element_type is None or element_type == 'text':
            tree = ET.parse(path)
            root = tree.getroot()
            tokens = root.find('Augmented_Tokens')
            if filter:
                tokens = [token.text for token in tokens]
                filtered_doc = self.filter_doc(tokens)
                return filtered_doc
            else:
                tags = ['NN', 'V', 'J']
                tokens = ' '.join([pattern.sub('', token.text) for token in tokens
                                   if check(token.get('treebank_pos'), tags)
                                   and token.get('sentence') != '0'
                                   and len(pattern.sub('', token.text)) > 1])
                return tokens

        # ecb+ annotations
        else:
            #this is a dictionary with a list of strings for every
            #ecb annotation (slot) in the document
            #comes with only alphanumeric chars
            elements = self.get_document_template(path)
            if element_type == 'event_trigger':
                return [term for term in [term_list for term_list in elements['ACTION']]]
            elif element_type == 'participant':
                return [term for term in [term_list for term_list in elements['HUM_PARTICIPANT'] + elements['NON_HUM_PARTICIPANT']]]
            elif element_type == 'time':
                return [term for term in [term_list for term_list in elements['TIME']]]
            elif element_type == 'location':
                return [term for term in [term_list for term_list in elements['LOCATION']]]
            elif element_type == 'doc_template':
                elements = elements['ACTION'] + elements['HUM_PARTICIPANT'] + elements['NON_HUM_PARTICIPANT'] + elements['LOCATION'] + elements['TIME']
                return [term for term in [term_list for term_list in elements]]
            elif element_type == 'event_hum_participant':
                return [term for term in [term_list for term_list in elements['ACTION'] + elements['HUM_PARTICIPANT']]]
            elif element_type == 'hum_participant':
                return [term for term in [term_list for term_list in elements['HUM_PARTICIPANT']]]
            elif element_type == 'event_hum_participant_location':
                return [term for term in [term_list for term_list in elements['ACTION'] + elements['HUM_PARTICIPANT'] + elements['LOCATION']]]

    '''
    Builds a document template for an ECB+ document. A document template includes all event components
    in a document. 
    
        input: 
        - path: path of desired ECB+ document
        - lemmatize: if you want to lemmatize elements in template or not
        
        output: 
        - a dictionary with key = slot, value = list of elements, with non-alphanumeric chars stripped
    '''
    def get_document_template(self,path):
        tree = ET.parse(path)
        root = tree.getroot()
        template = {"ACTION":[],
                    "HUM_PARTICIPANT":[],
                    "NON_HUM_PARTICIPANT":[],
                    "LOCATION":[],
                    "TIME":[]}
        lemmas = self.lemmatize_ecb_file(path)
        for markable in root.findall('Markables'):
            for child in markable:
                tags = [c.tag for c in child]
                if 'token_anchor' in tags:
                    if 'ACTION' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(self.only_alphanumeric(self.get_token(path,c.attrib['t_id']).text) for c in child)
                        template['ACTION'].append(txt)
                    elif 'PART' in child.tag:
                        if 'NON_HUM' in child.tag:
                            if self.lemmatize:
                                txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                            else:
                                txt = ' '.join(
                                    self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                            template['NON_HUM_PARTICIPANT'].append(txt)
                        else:
                            if self.lemmatize:
                                txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                            else:
                                txt = ' '.join(
                                    self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                            template['HUM_PARTICIPANT'].append(txt)
                    elif 'LOC' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(
                                self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                        template['LOCATION'].append(txt)
                    elif 'TIME' in child.tag:
                        if self.lemmatize:
                            txt = ' '.join(lemmas[c.attrib['t_id']] for c in child)
                        else:
                            txt = ' '.join(
                                self.only_alphanumeric(self.get_token(path, c.attrib['t_id']).text) for c in child)
                        template['TIME'].append(txt)
        return template

    '''
    Lemmatizes every word in an ECB+ file. 

        input: 
        - path: path of file you want to lemmatize 

        output: 
        - a dictionary where key = ECB+ token id (t_id), value = lowercase lemma, with non-alphanumeric chars stripped
    '''
    def lemmatize_ecb_file(self, path):
        lemmas = dict()
        lemmatizer = WordNetLemmatizer()
        for k, v in self.get_all_sentences(path).items():
            pos_tag = nltk.pos_tag([self.only_alphanumeric(t.text, keep_punct=True) for t in v])
            for i in range(len(pos_tag)):
                lemmas[v[i].attrib['t_id']] = (self.get_wordnet_pos(pos_tag[i][1]),
                                               lemmatizer.lemmatize(pos_tag[i][0],
                                                                    self.get_wordnet_pos(pos_tag[i][1])).lower())
        return lemmas

    '''
    Adds attributes to the ECB+ tokens
    
        input:
        - path: path of the file you want to augment
        - output_dir: where you want to store augmented file. file name will be existingName_aug.xml
        
        output: 
        - an edited ECB+ file, adding the following attributes to each token: 
            * event_type: ACTION_OCCURRENCE, etc., if token is part of a mention, '' if N/A
            * multi_token: span if event element is multi-token, as "k,...,n", '' if N/A
            * ev_id: ie. ACT234832941, corresponding to the ECB+ event id, '' if N/A
            * pos
            * lemma
            * wordnet_id
            * dbPedia_id
    '''
    def augment_ecb_tokens(self,path,output_dir):
        tree = ET.parse(path)
        root = tree.getroot()
        root.set('doc_name', root.attrib['doc_name'].replace('.xml', '_aug.xml'))
        augs = ET.SubElement(root,"Augmented_Tokens")

        lemmatizer = WordNetLemmatizer()
        #for all sentences
        for sent_num, tokens in self.get_all_sentences(path).items():
            #ecb+ is already tokenized
            pos_tag = nltk.pos_tag([t.text for t in tokens])
            context = [lemmatizer.lemmatize(tag[0],self.get_wordnet_pos(tag[1])) for tag in pos_tag]
            #for all tokens
            for i in range(len(pos_tag)):
                #lemma
                lemma = lemmatizer.lemmatize(pos_tag[i][0],self.get_wordnet_pos(pos_tag[i][1]).lower())

                #synset assignment
                syn = adapted_lesk(' '.join(context), ambiguous_word=tokens[i].text, pos=pos_tag[i][1],
                                   context_is_lemmatized=True)
                wordnet_id = ''
                if not syn is None:
                    wordnet_id = syn.name()

                #get event info
                ev_info = self.get_event_info(path,tokens[i].attrib['t_id'])
                ev_type = ev_id = ''
                if not ev_info is None:
                    ev_type = ev_info[0]
                    ev_id = ev_info[1]


                #set attributes
                tokens[i].set('ev_type', ev_type)
                tokens[i].set('ev_id', ev_id)
                tokens[i].set('treebank_pos',pos_tag[i][1])
                tokens[i].set('lemma',lemma)
                tokens[i].set('wordnet_id',wordnet_id)
                tokens[i].set('m_id',self.get_mention_id(path,tokens[i].attrib['t_id']))
                #tokens[i].set('dbPedia_id','')
                augs.append(tokens[i])

        tree.write(output_dir + '/' + root.attrib['doc_name'])
    '''
    Get event type and event id for a given t_id
        input:
            - path: path of the file you want to investigate
            - t_id: t_id of the token you want the m_id of
        output:
            - (ev_type,ev_id) tuple for given t_id in path
    '''
    def get_event_info(self,path,t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for markables in root.findall('Markables'):
            for event_element in markables:
                tags = [c.tag for c in event_element]
                if 'token_anchor' in tags:
                    m_id = event_element.attrib['m_id']
                    ev_type = event_element.tag
                    t_ids = [c.attrib['t_id'] for c in event_element]
                    #found the mention
                    if t_id in t_ids:
                        for relations in root.findall('Relations'):
                            for coref in relations:
                                m_ids = [s.attrib['m_id'] for s in coref.findall('source')]
                                #found ev_id
                                if m_id in m_ids:
                                    if'CROSS_DOC' in coref.tag:
                                        ev_id = coref.attrib['note']
                                        return (ev_type,ev_id)

    '''
    Returns the mention id (m_id) of the given token in the given document
        
        input:
            - path: path of the file you want to investigate
            - t_id: t_id of the token you want the m_id of
        output:
            - the m_id of the given t_id in documnent at path
        
    '''
    def get_mention_id(self,path,t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for markables in root.findall('Markables'):
            for event_element in markables:
                tags = [c.tag for c in event_element]
                if 'token_anchor' in tags:
                    m_id = event_element.attrib['m_id']
                    ev_type = event_element.tag
                    t_ids = [c.attrib['t_id'] for c in event_element]
                    #found the mention
                    if t_id in t_ids:
                        return m_id
        return ''

    '''
    Maps a Treebank tag to a Wordnet tag

        input: 
        - treebank_tag: Treebank tag to be mapped to Wordnet tag

        output: 
        - a wordnet object with the corresponding POS tag
    '''
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        #default to noun
        else:
            return wordnet.NOUN

    '''
    Returns an xml element representing the token in the path with given t_id
    
        input: 
        - path: path of desired document
        - t_id: token id (index, 1 based) of in ECB+ document
        
        output: 
        - xml element of that token 
    '''
    def get_token(self,path,t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for child in root.findall('token'):
            if child.attrib['t_id'] == str(t_id):
                return child

    '''
    Returns all sentences in an ECB+ document 
    
        input: 
        - path: path of desired document 
        
        output: 
        - dictionary with key = sentence id, value = list of tokens
    '''
    def get_all_sentences(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        sentences = dict()
        for child in root.findall('token'):
            if child.attrib['sentence'] not in sentences:
                sentences[child.attrib['sentence']] = []
            sentences[child.attrib['sentence']].append(child)
        return sentences

    '''
    Returns a sentence with a given sentence id
    
        input: 
        - path: path of desired ECB+ document
        - s_id: sentence id of desired sentence 
        
        output: 
        - list of tokens in that sentence
    '''
    def get_sentence(self,path,s_id):
        tree = ET.parse(path)
        root = tree.getroot()
        sentence = []
        for child in root.findall('token'):
            if child.attrib['sentence'] == str(s_id):
                sentence.append(child)
        return sentence

    '''
    Get xlm element representing that token 
    
        input: 
        - path: path of desired ECB+ document
        - t_id: token id
        
        output: 
        - xlm element of that token
    '''
    def get_token(self,path, t_id):
        tree = ET.parse(path)
        root = tree.getroot()
        for tok in root.findall('token'):
            if tok.attrib['t_id'] == t_id:
                return tok

    '''
    Computes a set of non-stopword words in corpus and saves in .txt file
    
        output:
        - set of terms, with non-alphanumeric chars stripped
    '''
    def compute_term_set(self):
        terms = set()
        for f in self.all_files:
            words = self.get_text(f)
            for w in words:
                terms.add(self.only_alphanumeric(w.lower()))
        with open('data/ecb_term_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms

    '''
    Computes set of event triggers (could be multi-word) and saves in .txt file
    
        output: 
        - set of event triggers
    '''
    def compute_event_trigger_set(self):
        terms = set()
        for f in self.all_files:
            triggers = self.get_text(path=f,element_type='event_trigger')
            for t in triggers:
                terms.add(self.only_alphanumeric(t))
        with open('data/ecb_event_trigger_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms

    '''
    Computes set of document template elements (could be multi word and contain non-alphanumeric characters)
    and saves in a .txt file
    
        output: 
        - set of document template elements
    '''
    def compute_doc_template_set(self):
        terms = set()
        for f in self.all_files:
            elements = self.get_document_template(f)
            for slot,term_list in elements.items():
                for t in term_list:
                    terms.add(self.only_alphanumeric(t))
        with open('data/ecb_doc_template_set.txt', 'w') as f:
            for item in terms:
                f.write("%s\n" % item)
        return terms
    '''
    Strip all non-alphanumeric chars from a string 
    
        input:
        - s: desired string
        
        output: 
        - same string w/o non-alphanumeric chars 
    '''
    def only_alphanumeric(self, s, keep_punct = False):
        if keep_punct:
            if len(s) > 1:
                return re.sub(r'\"', '', s)
            else:
                return s
        else:
            return re.sub(r'\W+', '', s)

