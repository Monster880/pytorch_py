import jieba

data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"
data_list = open(data_path,encoding='utf-8').readlines()[1:]
#处理停用词
stops_word = open(data_stop_path,encoding='utf-8').readlines()
stops_word = [line.strip() for line in stops_word]#去掉换行符
stops_word.append(" ")#防止被strip过滤掉
stops_word.append("\n")
#根据分词结果统计字典
voc_dict = {}
min_seq = 1
top_n = 1000
UNK="<UNK>"
PAD = "<PAD>"
for item in data_list[:]:
    label = item[0]
    content = item[2:].strip()#去掉结尾的换行符
    seg_list = jieba.cut(content, cut_all=False)
    seg_res = []

    for seg_item in seg_list:
        print(seg_item)
        if seg_item in stops_word: #去掉停用词
            continue
        seg_res.append(seg_item)
        if seg_item in voc_dict.keys(): #使用字典统计词频
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1

    print(content)
    print(seg_res)

#排序字典词频，取topN的词定义字典
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x:x[1], reverse=True)[:top_n]
voc_dict = {word_count[0]: idx for idx,word_count in enumerate(voc_list)}
#将字典以外的词固定为特定字符UNK
voc_dict.update({UNK:len(voc_dict),PAD:len(voc_dict) + 1})

print(voc_dict)

#保存字典
ff = open("sources/dict","w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item,voc_dict[item]))
