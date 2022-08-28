from ner import *


pre_model = torch.load('pre_model.pth')                            # 直接加载模型
with torch.no_grad():
    for i in range(10):
        test_data = data['train'][i]
        In = prepare_sequence(test_data[0],word_to_ix)
        Out = pre_model(In)
        print(f'模型预测输出: {Out}')
        targets = torch.tensor([tag_to_ix[tag] for tag in test_data[1]], dtype=torch.long)
        print(f'真值: {targets}')
        print(f'预测中词性标注错误的个数: {(torch.tensor(Out[1],dtype=torch.long)-targets).sum().item()}')
        print()
        # We got it!