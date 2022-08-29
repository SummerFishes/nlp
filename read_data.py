import pandas as pd
from tqdm import tqdm


def get_tags(label, length):
    tags = ['O'] * length
    for entity_type, [start, end] in label:
        entity_type = entity_type[0:3]
        tags[start] = 'B-' + entity_type
        for i in range(start+1, end):
            tags[i] = 'I-' + entity_type
    return tags


def read(file):
    df = pd.read_json(file)
    data_size = len(df)
    data_set = {'training': [], 'test': []}
    word_to_ix = {}
    for i in tqdm(range(0, data_size//5*4), desc='读取训练数据', position=0):
        text, label = df.loc[i]
        tokens = text.split()
        tags = get_tags(label, len(text))
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        data_set['training'].append((tokens, tags))
    for i in tqdm(range(data_size//5*4, data_size), desc='读取测试数据', position=0):
        text, label = df.loc[i]
        tokens = text.split()
        tags = get_tags(label, len(text))
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        data_set['test'].append((tokens, tags))
    print(len(data_set['training']))
    print(len(data_set['test']))
    return data_set, word_to_ix
