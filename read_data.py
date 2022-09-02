import pandas as pd
from tqdm import tqdm


# def get_tags(label, length):
#     tags = ['O'] * length
#     for entity_type, [start, end] in label:
#         entity_type = entity_type[0:3]
#         tags[start] = 'B-' + entity_type
#         for i in range(start+1, end)l:
#             tags[i] = 'I-' + entity_type
#     return tags


def get_tags(label, tokens):
    length = len(tokens)
    tags = ['O'] * length
    index = []*length
    last = 0
    for i in range(length):
        index.append(last+len(tokens[i]))
        last = index[-1]+1
#     print(index)
    label.sort(key=lambda x: x[1][0])
    help_map = {0:'B-', 1:'I-'}
#     print(label)
    label2 = [(l, x[0:3]) for x, y in label for l in y]
#     print(label2)
    i, j = 0, 0
    while i < length and j < len(label):    # 双指针
        if label[j][1][0] < index[i] <= label[j][1][1]:
            tags[i] = label[j][0][0:3]
        if index[i] > label[j][1][1]:
            j += 1
            i -= 1
        i += 1
    for i in range(length-1, -1, -1):
        if tags[i] != 'O':
            if i == 0 or tags[i-1] != tags[i]:
                tags[i] = f'B-{tags[i]}'
            else:
                tags[i] = f'I-{tags[i]}'
    return tags


def read(file):
    df = pd.read_json(file)
    data_size = len(df)
    data_set = {'training': [], 'test': []}
    word_to_ix = {'$unk#': 0}
    # word_to_ix = {}
    for i in tqdm(range(0, data_size//5*4), desc='读取训练数据', position=0):
        text, label = df.loc[i]
        tokens = text.split()
        tags = get_tags(label, tokens)
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        data_set['training'].append((tokens, tags))
    for i in tqdm(range(data_size//5*4, data_size), desc='读取测试数据', position=0):
        text, label = df.loc[i]
        tokens = text.split()
        tags = get_tags(label, tokens)
        for word in tokens:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        data_set['test'].append((tokens, tags))
    # print(len(data_set['training']))
    # print(len(data_set['test']))
    return data_set, word_to_ix
