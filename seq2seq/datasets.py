import jieba
from utils import normalizeString#字符串处理
from utils import cht_to_chs#繁体字转简体字
#提取出中文和英文作为语料对 统计字典

SOS_token = 0 #起始符和终止符
EOS_token = 1
MAX_LENGTH = 10

class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {} #对词语进行编码
        self.word2cont = {} #统计字典中每个词出现的频率
        self.index2word = { #索引对应的词
                0:"SOS", 1:"EOS"
        }
        self.n_words = 2 #统计当前语料库中的单词数目

    #用来对词进行统计，利用word更新字典值
    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words#给每个词一个索引值
            self.word2cont[word] = 1
            self.index2word[self.n_words] = word
            self.n_words +=1
        else:
            self.word2cont[word] +=1

    #用来分词
    def addSentence(self,sentence):
        for word in sentence.split(" "):
            self.addWord(word)

#文本解析
def readLangs(lang1, lang2, path):
    lines = open(path,encoding="utf-8").readlines()

    lang1_cls = Lang(lang1)
    lang2_cls = Lang(lang2)

    pairs = []
    for l in lines:
        l = l.split("\t")
        sentence1 = normalizeString(l[0]) #英文
        sentence2 = cht_to_chs(l[1]) #中文
        seg_list = jieba.cut(sentence2, cut_all=False) #分词结果
        sentence2 = " ".join(seg_list) #通过空格拼接分词结果

        if len(sentence1.split(" ")) > MAX_LENGTH: #过滤长句子
            continue
        if len(sentence2.split(" ")) > MAX_LENGTH:
            continue

        pairs.append([sentence1, sentence2])
        lang1_cls.addSentence(sentence1)
        lang2_cls.addSentence(sentence2)

    return lang1_cls, lang2_cls, pairs

lang1 = "en"
lang2 = "cn"
path = "data/en-cn.txt"
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path)

print(len(pairs))
print(lang1_cls.n_words)
print(lang1_cls.index2word)

print(lang2_cls.n_words)
print(lang2_cls.index2word)
