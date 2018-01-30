import json
import pandas as pd
from tqdm import tqdm


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


if __name__ == '__main__':
    standardize_squad('data/train-v1.1.json')
    standardize_squad('data/dev-v1.1.json')
