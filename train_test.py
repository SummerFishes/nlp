import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
from tqdm.notebook import trange
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from bilstm_crf import *
from read_data import read

epoch_times = 300
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5  # 词嵌入维度
HIDDEN_DIM = 4  # 隐层层数
tag_to_ix = {  # 标签词典 {标注——索引}
    "B-NAM": 0, "B-TIC": 1, "B-NOT": 2,
    "I-NAM": 3, "I-TIC": 4, "I-NOT": 5,
    "O": 6, START_TAG: 7, STOP_TAG: 8
}
data_set, word_to_ix = read('./data/data.json')  # 数据集，词典 {词——索引}


def compute_loss(pre, target):
    loss_func = torch.nn.SmoothL1Loss()
    print([(x, y) for x, y in zip(pre, target)][0])
    loss = [loss_func(x, y) for x, y in zip(pre, target)]
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # 设置图像显示大小
    plt.rcParams['image.interpolation'] = 'nearest'  # 设置差值方式
    plt.rcParams['image.cmap'] = 'gray'  # 设置灰度空间
    plt.plot(np.squeeze(loss))
    plt.ylabel('loss')
    plt.xlabel('iterations (per tens)')
    plt.title(f"Learning rate = 0.01, epoch_times = {epoch_times}")
    plt.show()


def add_batch(data_input, data_label, batch_size):
    """
    设置batch
    :param data_input: 输入
    :param data_label: 标签
    :param batch_size: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(torch.Tensor(data_input), torch.Tensor(data_label))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)#shuffle是是否打乱数据集，可自行设置

    return data_loader


def train():
    training_data = data_set['training']

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    data_in, data_label = [], []
    # for tokens, tags in training_data:
    #     data_in.append(tokens)
    #     data_label.append(tags)
    # training_batch = add_batch(data_in, data_label, 10)
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in trange(300, desc='模型训练进度'):
        bar = tqdm(enumerate(training_data), leave=False)
        for step, (sentence, tags) in bar:
            bar.set_description(f'epoch【{epoch}】')
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags],
                                   dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                state = {
                    'epoch' : epoch + 1,                   # 保存当前的迭代次数
                    'state_dict' : model.state_dict(),     # 保存模型参数
                    'optimizer' : optimizer.state_dict(),  # 保存优化器参数
                }
                torch.save(state, f'./checkpoint2/checkpoint{epoch}.pth.tar')
    print('training over!')
    torch.save(model, 'pre_model.pth')
    torch.save(model.state_dict(), 'model_params.pth')


def test():
    test_data = data_set['test']
    length = len(test_data)
    # checkpoint = torch.load(f'./checkpoint/checkpoint{epoch_times}.pth.tar')
    # pre_model = BiLSTM_CRF(len(word_to_ix),tag_to_ix,EMBEDDING_DIM,HIDDEN_DIM)
    # pre_model.load_state_dict(checkpoint['state_dict'])
    pre_model = torch.load('pre_model.pth')  # 直接加载模型
    pre = []
    targets = []
    with torch.no_grad():
        for i in range(length):
            tokens, tags = test_data[i]
            # print(test_data[i])
            # print(' '.join(tokens), end=' ')
            # print(tags)
            model_in = prepare_sequence(tokens, word_to_ix)
            model_out = pre_model(model_in)
            # print(model_in, end=' ')
            # print(model_out)
            # print(f'模型预测输出: {model_out}')
            target = torch.tensor([tag_to_ix[tag] for tag in tags],
                                  dtype=torch.long)
            pre.append(torch.tensor(model_out[1]), dtype=torch.long)
            targets.append(target)
    # print(targets)
    compute_loss(pre, targets)


if __name__ == '__main__':
    test()
