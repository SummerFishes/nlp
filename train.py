from ner import *
from read_data import read

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5       # 词嵌入维度
HIDDEN_DIM = 4          # 隐层层数

data_set, word_to_ix = read('./data/data.json')  # 数据集，词典 {词——索引}
training_data = data_set['training']

tag_to_ix = {"B-NAM": 0, "B-TIC": 1, "B-NOT": 2,
             "I-NAM": 3, "I-TIC": 4, "I-NOT": 5,
             "O": 6, START_TAG: 7, STOP_TAG: 8}  # 标签词典 {标注——索引}
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in trange(300,desc='模型训练进度'):
    bar = tqdm(training_data, leave=False)
    for sentence, tags in bar:
        bar.set_description(f'epoch【{epoch}】')
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     print(model(precheck_sent))
print('traning over!')
torch.save(model,'pre_model.pth')
torch.save(model.state_dict(),'model_params.pth')