import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm, trange


torch.manual_seed(1)    # 人工设定随机种子以保证相同的初始化参数，实现模型的可复现性。
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):  # 给定输入二维序列，取每行（第一维度）的最大值，返回对应索引。
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):    # 利用to_ix这个word2id字典，将序列seq中的词转化为数字表示，包装为torch.long后返回
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):                # 函数目的相当于log∑exi 首先取序列中最大值，输入序列是一个二维序列(shape[1,tags_size])。下面的计算先将每个值减去最大值，再取log_sum_exp，最后加上最大值。
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 词嵌入维度，即输入维度
        self.hidden_dim = hidden_dim  # 隐层维度
        self.vocab_size = vocab_size  # 训练集词典大小
        self.tag_to_ix = tag_to_ix  # 标签索引表
        self.tagset_size = len(tag_to_ix)  # 标注 类型数
        print(f'tagset_size={self.tagset_size}')
        self.word_embeds = nn.Embedding(vocab_size,
                                        embedding_dim)  # （词嵌入的个数，嵌入维度）
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2,
                            # （输入节点数，隐层节点数，隐层层数，是否双向）
                            num_layers=1,
                            bidirectional=True)  # hidden_size除以2是为了使BiLSTM
        # 的输出维度依然是hidden_size,而不用乘以2

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim,
                                    self.tagset_size)  # （输入x的维度，输出y的维度），将LSTM的输出线性映射到标签空间

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(  # 转移矩阵，标注j转移到标注i的概率，后期要学习更新
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # 不会有标注转移到开始标注
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # 结束标注不会转移到其他标注

        self.hidden = self.init_hidden()

    def init_hidden(self):  # 初始化隐层（两层，3维）
        return (torch.randn(2, 1, self.hidden_dim//2),
            # (num_layer * num_direction, batch_size)
        torch.randn(2, 1, self.hidden_dim//2))  # (隐层层数2 * 方向数1， 批大小1， 每层节点数)

    def _forward_alg(self, feats):  # 得到所有路径的分数/概率
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size),
                                 -10000.)  # P，(1, m)维，初始化为-10000
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # 前向状态，记录当前t之前的所有路径的分数

        # Iterate through the sentence
        for feat in feats:  # 动态规划思想，具体见onenote上的笔记
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1,
                                                               self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var+trans_score+emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var+self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha  # 返回的是所有路径的分数

    def _get_lstm_features(self, sentence):  # 通过BiLSTM层，输出得到发射分数
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1,
                                                 -1)  # 对输入语句 词嵌入化
        lstm_out, self.hidden = self.lstm(embeds,
                                          self.hidden)  # 词嵌入通过lstm网络输出,lstm传入参数之后会自动调用其forward方法
        lstm_out = lstm_out.view(len(sentence),
                                 self.hidden_dim)  # 将输出转为2维（原本是3维，但是batch_size=1，可以去掉这一维）
        lstm_feats = self.hidden2tag(lstm_out)  # 将输出映射到标签空间，得到单词-分数表
        return lstm_feats

    def _score_sentence(self, feats, tags):  # 计算给定路径的分数
        # feats : LSTM的所有输出，发射分数矩阵
        # tags : golden路径的标注序列
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long),
                tags])  # 在标注序列最前加上开始标注
        for i, feat in enumerate(feats):  # 计算给定序列的分数，Σ发散分数+Σ转移分数
            score = score+self.transitions[tags[i+1], tags[i]]+feat[tags[i+1]]
        score = score+self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size),
                                -10000.)  # 初始化forward_var,并且 开始标注 的分数为0,确保一定是从START_TAG开始的,
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars  # forward_var记录每个标签的前向状态得分，即w{i-1}被打作每个标签的对应得分值
        for feat in feats:  # feats是LSTM的输出，每一个feat都是一个词w{i}，feat[tag]就是这个词tag标注的分数
            bptrs_t = []  # holds the backpointers for this step                     # 记录当前词w{i}对应每个标签的最优转移结点
            viterbivars_t = []  # holds the viterbi variables for this step          # 记录当前词各个标签w{i, j}对应的最高得分
            # 动态规划：w{i，j}=max{forwar_var + transitions[j]}，词存于bptrs_t中，分数存于viterbivars_t中

            for next_tag in range(self.tagset_size):  # 对当前词w{i}的每个标签 运算
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var+self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t)+feat).view(1, -1)
            backpointers.append(bptrs_t)  # 记忆，方便回溯

        # Transition to STOP_TAG
        terminal_var = forward_var+self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)  # 结束标记前的一个词的最高前向状态得分就是最优序列尾
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 回溯
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence,
                           tags):  # CRF的损失函数：-gold分数-logsumexp(所有序列)
        feats = self._get_lstm_features(
            sentence)  # 通过BiLSTM层，获得每个 {词-标签}对 的发射分数
        forward_score = self._forward_alg(feats)  # 根据发射分数计算所有路径的分数
        gold_score = self._score_sentence(feats,
                                          tags)  # 传入标注序列真实值，计算语句的真实分数gold_score
        return forward_score-gold_score  # 返回误差值

    def forward(self, sentence):  # 重载前向传播函数，对象传入参数后就会自动调用该函数
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)  # 通过LSTM层得到输出

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)  # 通过CFR层得到最优路径及其分数
        return score, tag_seq