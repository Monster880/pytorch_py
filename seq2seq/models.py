import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import MAX_LENGTH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#编码模块
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)#输入句子的长度，输出编码成词向量的长度
        self.gru = nn.GRU(hidden_size, hidden_size)#词向量长度 也可以选择lstm

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)#将编码后的维度传入到最后一个维度上去
        output = embedded
        output, hidden = self.gru(output,hidden)
        return output,hidden #返回gru输出的结果和隐藏层信息

    def initHidden(self): #在第一个节点上定义一个默认的隐藏层
        return torch.zeros(1, 1, self.hidden_size, device=device)

#解码模块 1.不带attention的结构 2.带attention的结构
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input, hidden):
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden #类别的概率分布和隐藏层信息

    def initHidden(self): #在第一个节点上定义一个默认的隐藏层
        return torch.zeros(1, 1, self.hidden_size, device=device)

#基于attention的解码结构
class AttenDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_len=MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_len = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)#要对两个结果进行连接，因此要乘以2
        self.attn_combine = nn.Linear(self.hidden_size *2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(self.hidden_size,self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)#对输入进行特征抽取转化为词向量
        embedded = self.dropout(embedded)

        atten_weight = F.softmax(
            self.attn(torch.cat([embedded[0], hidden[0]], 1)), #将embedded和hidden进行拼接，来学习attention权重
            dim=1
        )

        att_applied = torch.bmm( #计算两个tensor的矩阵乘法
            atten_weight.unsqueeze(0),#取出权重
            encoder_outputs.unsqueeze(0)
        ) #将权重作用到feature上

        output = torch.cat([embedded[0],att_applied[0]], dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)#加入非线性信息
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, atten_weight

    def initHidden(self): #在第一个节点上定义一个默认的隐藏层
        return torch.zeros(1, 1, self.hidden_size, device=device)

if __name__ == '__main__':
    encoder_net = EncoderRNN(5000, 256)
    decoder_net = DecoderRNN(256, 5000)
    atten_decoder_net = AttenDecoderRNN(256, 5000)

    tensor_in = torch.tensor([12, 14, 16, 18], dtype=torch.long).view(-1, 1)#定义输入并调整shape
    hidden_in = torch.zeros(1, 1, 256)
    #测试编码网络
    encoder_out, encoder_hidden = encoder_net(tensor_in[0], hidden_in)
    print(encoder_out)
    print(encoder_hidden)

    #测试解码网络
    tensor_in = torch.tensor([100])
    hidden_in = torch.zeros(1, 1, 256)
    encoder_out = torch.zeros(10,256) #第一维大小取决于MAX_LENGTH,此处为10

    out1, out2, out3 = atten_decoder_net(tensor_in, hidden_in, encoder_out)
    print(out1, out2, out3)

    out1, out2 = decoder_net(tensor_in,hidden_in)
    print(out1,out2)
