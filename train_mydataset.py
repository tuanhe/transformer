import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from models.model.transformer import Transformer
import argparse
#device = 'cpu'
device = 'cuda'

# transformer epochs
epochs = 100
# epochs = 1000

training_sentences = [
    # 中文和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
]

# 中文和英语的单词要分开建立词库
# Padding Should be Zero
source_vocabulary = {'P': 0, '我': 1, '有': 2, '一': 3, '个': 4, '好': 5,
             '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
source_idx2word = {i: w for i, w in enumerate(source_vocabulary)}
source_vocabulary_size = len(source_vocabulary)

target_vocabulary = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4, 'friend': 5, 
                     'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11}
target_idx2word = {i: w for i, w in enumerate(target_vocabulary)}
target_vocabulary_size = len(target_vocabulary)

ffn_hidden = 2048
d_model = 512  # Embedding Size（token embedding和position编码的维度）
n_block = 6
n_heads = 8
max_len = 5000
# ==============================================================================================
# 数据构建

def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
 
        enc_input = [[source_vocabulary[n] for n in sentences[i][0].split()]]
        dec_input = [[target_vocabulary[n] for n in sentences[i][1].split()]]
        dec_output = [[target_vocabulary[n] for n in sentences[i][2].split()]]

        #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.extend(enc_input)
        #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input)
        #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trainning(training_dataset):
    print("Data training")

    enc_inputs, dec_inputs, dec_outputs = make_data(training_dataset)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    src_pad_idx = target_vocabulary['P']
    trg_pad_idx = target_vocabulary['P']
    trg_sos_idx = target_vocabulary['S']

    #model = Transformer().to(device)
    model = Transformer(src_pad_idx = src_pad_idx,
                        trg_pad_idx = trg_pad_idx,
                        trg_sos_idx = trg_sos_idx,
                        d_model = d_model,
                        enc_voc_size = source_vocabulary_size,
                        dec_voc_size = target_vocabulary_size,
                        max_len = max_len,
                        ffn_hidden = ffn_hidden,
                        n_head = n_heads,
                        n_layers = n_block,
                        drop_prob = 0.1,
                        device=device).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                          momentum=0.99)  # 用adam的话效果不好

    # ====================================================================================================
    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)
            # outputs: [batch_size * tgt_len, target_vocabulary_size]
            # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # dec_outputs.view(-1):[batch_size * tgt_len * target_vocabulary_size]
            # print("check outputs shape : ", outputs.shape)
            # print("check decoder outputs shape : ", dec_outputs.shape)
            # print("check shape : ", dec_outputs.view(-1).shape)
            # print("\n")
            # loss = criterion(outputs, dec_outputs.view(-1))
            
            outputs = model(enc_inputs, dec_inputs)
            reshaped_output = outputs.view(-1, outputs.size(-1))
            loss = criterion(reshaped_output, dec_outputs.view(-1))
            '''
            outputs = model(enc_inputs, dec_inputs)

            output_reshape = outputs.contiguous().view(-1, outputs.shape[-1])
            dec_outputs = dec_outputs[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, dec_outputs)
            '''

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    print("Training done")
    return model

def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    # [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_self_attn_mask = get_attn_pad_mask(enc_input, enc_input) 
    enc_outputs = model.encoder(enc_input, enc_self_attn_mask)
    print("enc_outputs:", enc_outputs)
    print("enc_outputs shape:", enc_outputs.shape)
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    print("Inference dec_input:" , dec_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = dec_input.to(device)
        # 创建一个形状为(1, 1)的张量，其中包含了一个整数next_symbol，并将其移动到指定的设备上
        next_symbol_tensor = torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)
        # 使用torch.cat函数将两个张量连接起来，得到一个新的张量
        dec_input = torch.cat([dec_input, next_symbol_tensor], dim=-1)
        print(dec_input)
        
        dec_enc_attn_mask = model.make_src_mask(enc_input)
        dec_self_attn_mask = model.make_trg_mask(dec_input)
        #dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
        output = model.decoder(dec_input, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        
        print("Output shape is \n", output.shape)

        # reshaped_output = output.view(-1, output.size(-1))
        #projected = model.projection(dec_outputs)
        prob = output.squeeze(0).max(dim=-1, keepdim=False)[1]

        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测是会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == target_vocabulary["E"]:
            terminal = True
        # print(next_word)

    # greedy_dec_predict = torch.cat(
    #     [dec_input.to(device), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
    #     -1)
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


# ==========================================================================================
# 预测阶段
# 测试集
inference_sentences = [
    # enc_input                dec_input           dec_output
    ['我 有 零 个 女 朋 友 P', '', ''],
    ['我 有 一 个 女 朋 友 P', '', '']
]

def inference(mode, inference_data):
    enc_inputs, dec_inputs, dec_outputs = make_data(inference_data)
    test_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    enc_inputs, _, _ = next(iter(test_loader))

    print()
    print("="*30)
    print("利用训练好的Transformer模型将中文句子'我 有 零 个 女 朋 友' 翻译成英文句子: ")
    for i in range(len(enc_inputs)):
        greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(
            1, -1).to(device), start_symbol=target_vocabulary["S"])
        print("greedy_dec_predict:\n", greedy_dec_predict)
        print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
        print([source_idx2word[t.item()] for t in enc_inputs[i]], '->',
              [target_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--argument', type=str, default='default_value', help='an example argument')
    args = parser.parse_args()
    model = trainning(training_sentences)
    torch.save(model, 'saved/model.pth')
    inference(model, inference_sentences)