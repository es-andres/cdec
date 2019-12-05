import xml.etree.ElementTree as ET
import os
import shared_vars as vars
import subprocess
import sys
import copy

cleaned = dict()
with open(vars.CLEAN_SENT_PATH) as f:
    for line in f.readlines()[1:]:
        line = line.split(',')
        f_name = line[0] + '_' + line[1]
        f_name = str(f_name + '_aug.xml')
        if f_name not in cleaned:
            cleaned[f_name] = []
        cleaned[f_name].append(line[2].replace('\n', ''))
total_sents = 0
for f in cleaned:
    total_sents += len(cleaned[f])
print(total_sents, 'total annotated sentences in', len(cleaned), 'files')


class ECBDocWrapper:

    def __init__(self, path):
        self.path = path
        self.fname = os.path.basename(self.path)
        self.aug_fname = self.fname.replace('.xml', '_aug.xml') if 'aug' not in self.fname else self.fname
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()

    def augment_ecb_tokens(self):
        self.root.set('doc_name', self.root.attrib['doc_name'].replace('.xml', '_aug.xml'))
        aug_toks = ET.SubElement(self.root, "Augmented_Tokens")
        pred_evs = dict()
        if self.aug_fname in cleaned:
            pred_evs = self.predict_events()
        for tok in self.get_all_tokens():
            aug_tok = copy.deepcopy(tok)
            s_id = aug_tok.attrib['sentence']
            if s_id in pred_evs and str(int(aug_tok.attrib['number']) + 1) in pred_evs[s_id]:
                aug_tok.set('pred_ev', pred_evs[s_id][str(int(aug_tok.attrib['number']) + 1)])
            else:
                aug_tok.set('pred_ev', '')
            m_id, ev_type = self.get_mention_info(aug_tok.attrib['t_id'])
            ev_id = self.get_ev_id(m_id)

            aug_tok.set('ev_type', ev_type)
            aug_tok.set('ev_id', ev_id)
            aug_tok.set('m_id', m_id)
            aug_tok.tag = 'aug_token'
            aug_toks.append(aug_tok)

        self.tree.write(os.path.join(vars.ECB_AUG_DIR, self.root.attrib['doc_name']))

    def calculate_ev_pred_performance(self):
        aug_toks = self.root.find('Augmented_Tokens')
        res = {'tp': 0,
               'fp': 0,
               'tn': 0,
               'fn': 0}
        if self.aug_fname not in cleaned:
            return res
        for tok in aug_toks:
            if tok.attrib['sentence'] in cleaned[self.aug_fname]:
                pred = tok.attrib['pred_ev'] != ''
                label = tok.attrib['ev_id'].startswith('ACT')
                if pred and label:
                    res['tp'] += 1
                if pred and not label:
                    res['fp'] += 1
                if not pred and not label:
                    res['tn'] += 1
                if not pred and label:
                    res['fn'] += 1
        return res

    def count_evs_and_chains(self):
        num_evs = 0
        chains = dict()
        aug_toks = self.root.find('Augmented_Tokens')
        if self.aug_fname not in cleaned:
            return num_evs, chains
        for tok in aug_toks:
            if tok.attrib['sentence'] in cleaned[self.aug_fname]:
                if tok.attrib['ev_id'].startswith('ACT'):
                    num_evs += 1
                    if tok.attrib['ev_id'] not in chains:
                        chains[tok.attrib['ev_id']] = []
                    chains[tok.attrib['ev_id']] = list(set(chains[tok.attrib['ev_id']] + [self.fname + '_' + tok.attrib['m_id']]))
        return num_evs, chains



    def predict_events(self):
        s_id_to_offsets = dict()
        sentences = dict()
        for tok in self.get_all_tokens():
            if tok.attrib['sentence'] not in sentences:
                sentences[tok.attrib['sentence']] = []
            sentences[tok.attrib['sentence']].append(tok.text)
        for s_id in sentences:
            if s_id not in cleaned[self.aug_fname]:
                continue
            txt = ' '.join(sentences[s_id])
            with open(vars.TXT_PATH, "w") as text_file:
                print(f"{txt}", file=text_file)
            subprocess.run(vars.CAEVO_ARGS, cwd=vars.CAEVO_PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tree = ET.parse(vars.CAEVO_OUTPUT)
            root = tree.getroot()[0][0]
            for child in root:
                if 'events' in child.tag:
                    for ev in child:
                        if s_id not in s_id_to_offsets:
                            s_id_to_offsets[s_id] = dict()
                        s_id_to_offsets[s_id][ev.attrib['offset']] = 'pred_' + s_id + '_' + ev.attrib['id']

        return s_id_to_offsets

    def get_all_tokens(self):
        toks = []
        for child in self.root.findall('token'):
            toks.append(child)
        return toks

    '''
    Get event type and event id for a given t_id
        input:
            - path: path of the file you want to investigate
            - t_id: t_id of the token you want the m_id of
        output:
            - (ev_type,ev_id) tuple for given t_id in path
    '''
    def get_ev_id(self, m_id):
        if m_id == '':
            return ''

        for relations in self.root.findall('Relations'):
            for coref_tag in relations:
                m_ids = [s.attrib['m_id'] for s in coref_tag.findall('source')] # 'source' refers to current doc.
                if m_id in m_ids and 'CROSS_DOC' in coref_tag.tag:
                    return coref_tag.attrib['note']
        return ''

    '''
    Returns the mention id (m_id) of the given token in the given document

        input:
            - path: path of the file you want to investigate
            - t_id: t_id of the token you want the m_id of
        output:
            - the m_id of the given t_id in documnent at path

    '''
    def get_mention_info(self, t_id):
        for markables in self.root.findall('Markables'):
            for event_element in markables:
                tags = [c.tag for c in event_element]
                if 'token_anchor' in tags:
                    m_id = event_element.attrib['m_id']
                    ev_type = event_element.tag
                    t_ids = [c.attrib['t_id'] for c in event_element]
                    # found the mention
                    if t_id in t_ids:
                        return m_id, ev_type
        return '', ''


