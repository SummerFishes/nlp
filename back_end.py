from collections import defaultdict

import torch
from flask import Flask
from flask import request
from flask_cors import CORS

from bilstm_crf import prepare_sequence
from read_data import read

app = Flask(__name__)  # 如果是单独应用可以使用__name__，如果是module则用模块名
# Flask还有其他参数https://blog.csdn.net/YZL40514131/article/details/122730037
CORS(app, supports_credentials=True)  # 解决跨域


@app.errorhandler(404)  # 8.定义错误处理的方法，参数是状态码
def handle_404_error(err):
    """自定义的处理错误方法"""
    # 这个函数的返回值会是前端用户看到的最终结果
    return "出现了404错误， 错误信息：%s" % err


@app.route('/get')  # 9.get请求
def get_test():
    data = 'test'
    return {'data': data}


@app.route('/nlp', methods=["POST"])  # 10.post请求
# 获取参数看content-type,见【https://blog.csdn.net/ling620/article/details/107562294】
def nlp():
    print('nlp start')
    print(request.content_type)
    print(request.json)
    sentence = request.json.get('sentence')
    pre_model = torch.load('pre_model.pth')  # 直接加载模型
    data_set, word_to_ix = read('./data/data.json')
    with torch.no_grad():
        tokens = sentence.split()
        model_in = prepare_sequence(tokens, word_to_ix)
        model_out = pre_model(model_in)
        nlp_result = get_entity(model_out[1], sentence)
    # nlp_result = {
    #     "NAME": "",
    #     "TICKER": "",
    #     "NOTIONAL": ""
    # }
    return nlp_result


def get_entity(ix_seq, sentence):
    ix_to_tag = {0: "NAME", 1: "TICKER", 2: "NOTIONAL", }
    # tag_seq = [ix_to_tag[ix] for ix in ix_seq]
    entities = defaultdict(str)
    i = 0
    while i < len(ix_seq):
        ix = ix_seq[i]
        if ix in ix_to_tag:
            j = i+1
            while j < len(ix_seq) and ix_seq[j] == ix+3:
                j += 1
            entities[ix_to_tag[ix]] = sentence[i: j]
            i = j-1
        i += 1
    return entities


if __name__ == '__main__':
    app.run()
