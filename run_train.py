import torch
import torch.nn as nn
from torch import optim
from models import Model
from dataset import data_loader, text_Cls
import configs

cfg = configs.Config()

#读取数据
data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword.txt"
dict_path = "sources/dict"

dataset = text_Cls(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(data_path,data_stop_path, dict_path)

cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_text_cls.parameters(),lr=cfg.learn_rate)

for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.devices)
        label = torch.tensor(label,dtype=torch.int64).to(cfg.devices)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        # print(pred)
        # print(label)

        print("epoch is {}, ite is {}, val is {}".format(epoch,i,loss_val))
        loss_val.backward()
        optimizer.step()

    if epoch % 10 == 0:#每10次迭代存储一次模型
        torch.save(model_text_cls.state_dict(),"models/{}.pth".format(epoch))


