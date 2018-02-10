import json
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm


nlp = spacy.load('en')


def standardize_squad(fname):
    with open(fname, 'r') as fl:
        squad = json.load(fl)

    rows, desc = [], 'Building DF ' + fname
    for docid, doc in enumerate(tqdm(squad['data'],
                                     desc=desc)):
        for para in doc['paragraphs']:
            context = para['context']
            for qas in para['qas']:
                qid = qas['id']
                q = qas['question']
                a = qas['answers'][0]['text']
                s = qas['answers'][0]['answer_start']
                e = s + len(a)
                rows.append([qid, docid, context, q, a, s, e])
    df = pd.DataFrame(rows)
    df.columns = ['qid', 'docid', 'context', 'question', 'answer',
                  'start', 'end']
    return df


def ohe(i, m):
    v = [0] * m
    if i < m:
        v[i] = 1
    return v


def load_squad(fname):
    df = standardize_squad(fname)
    para_toks = {c: [i.text for i in nlp.tokenizer(c)]
                 for c in tqdm(set(df['context']), desc=fname + ' Tok-C')}
    df['c_tokens'] = df['context'].map(para_toks)
    q_tokens = {q: [i.text for i in nlp.tokenizer(q)]
                for q in tqdm(set(df['question']), desc=fname + ' Tok-Q')}
    df['q_tokens'] = df['question'].map(q_tokens)
    df['start_exp_one_hot'] = list(df['start'])
    df['end_exp_one_hot'] = list(df['end'])
    return df


def load_glove(fname):
    glove = {}
    with open(fname, 'r') as fl:
        for line in tqdm(fl.readlines(), desc='Load GloVe'):
            w, v = line.split(' ', 1)
            w = w.strip() if w.strip() else ' '
            glove[w] = list(map(float, v.strip().split(' ')))
    return glove


def get_answer(s_prediction, e_prediction, batch):
    "Given predictions get answers"
    s_indices = np.argmax(s_prediction, axis=-1)
    e_indices = np.argmax(e_prediction, axis=-1)
    answers = {}
    ids, contexts = batch['qid'], batch['c_tokens']
    for id, con, s, e in zip(ids,
                             contexts,
                             s_indices,
                             e_indices):
        e = max(s, e)
        answers[id] = ' '.join([str(i) for i in con[s:e+1]])
    return answers


if __name__ == '__main__':
    standardize_squad('data/train-v1.1.json')
    standardize_squad('data/dev-v1.1.json')
