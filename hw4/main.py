import jieba
import sys
import math
import torch
from torch import nn
import time
import random


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


def data_iter_consecutive(corpus_index, batch_size, num_step, device=None):
    corpus_index = torch.tensor(
        corpus_index, dtype=torch.float32, device=device)
    data_len = len(corpus_index)
    batch_len = data_len // batch_size
    indices = corpus_index[0: batch_size *
                              batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_step
    for i in range(epoch_size):
        i = i * num_step
        X = indices[:, i: i + num_step]
        Y = indices[:, i + 1: i + num_step + 1]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()  # long() 函数将数字或字符串转换为一个长整型.
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    # print(x.view(-1, 1).shape)
    res.scatter_(1, x.view(-1, 1), 1)
    # 在res中，将1，按照dim=1(即不改行改列)的方向，根据[[0],[2]]所指示的位置，放入res中。（比如，x中的0，代表要放入第0列；而0本身处于第0行，所以是第0行中的第0列。）
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                        char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hidden, vocabulary_num, device,
                                  corpus_index, idx_to_char, char_to_idx,
                                  num_epoch, num_step, lr, clipping_theta,
                                  batch_size, predict_period, predict_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epoch):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_index, batch_size, num_step, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)

            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()

            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % predict_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, predict_len, model, vocabulary_num, device, idx_to_char,
                    char_to_idx))


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * \
                           (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size)  # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


if __name__ == '__main__':
    sys.path.append("..")
    device = torch.device('cpu')
    f = open('Corpus.txt', encoding='utf-8')
    corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0: 500000]
    corpus_chars = corpus_chars = jieba.lcut(corpus_chars)
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocabulary_num = len(char_to_idx)
    print(vocabulary_num)
    corpus_idex = [char_to_idx[char] for char in corpus_chars]
    num_input, num_hidden, num_output = vocabulary_num, 256, vocabulary_num
    num_epoch, num_step, batch_size, lr, clipping_theta = 200, 100, 256, 1e-2, 1e-2
    predict_period, predict_len, prefixes = 50, 50, ['王掌柜']
    lstm_layer = nn.LSTM(input_size=vocabulary_num, hidden_size=num_hidden, num_layers=1)
    model = RNNModel(lstm_layer, vocabulary_num)
    train_and_predict_rnn_pytorch(model, num_hidden, vocabulary_num, device, corpus_idex, idx_to_char,
                                  char_to_idx, num_epoch, num_step, lr, clipping_theta, batch_size,
                                  predict_period, predict_len, prefixes)
