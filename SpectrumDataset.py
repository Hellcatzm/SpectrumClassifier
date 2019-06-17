import pickle
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


f = open('fits_totul.pkl','rb')
data = pickle.load(f)

cla = [k.split("=")[-1] for k in data['subclass_raw_dict']]
data_dict = {i: data['subclass_raw'] == i
             for i in range(len(cla))}  # 标签：数据索引
X_data = data['fluxes_standard'].reshape(-1, 2000, 1, 1)  # 数据序列
y_data = data['subclass_raw']  # 数据标签
y_clcs = len(data['subclass_raw_dict'])  # 标签数目

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2)
train_dict = {i: y_train == i for i in range(len(cla))}
test_dict = {i: y_test == i for i in range(len(cla))}


class SpectrumDataset(Dataset):
    def __init__(self,
                 X=X_data,
                 y=y_data,
                 data_dict=data_dict,
                 y_clcs=y_clcs):

        self.X = X
        self.y = y
        self.data_dict = data_dict
        self.y_clcs = y_clcs
        self.labels = list(range(y_clcs))
        self.data_len = len(X)

    def __getitem__(self, index):
        label = self.labels[index]

        pos_datas = self.X[self.data_dict[label]]
        pos_index1, pos_index2 = np.random.choice(np.arange(len(pos_datas)), 2)
        pos_data1, pos_data2 = pos_datas[pos_index1], pos_datas[pos_index2]

        neg_label = np.random.choice(list(set(self.labels) ^ set([label])))
        neg_datas = self.X[self.data_dict[neg_label]]
        neg_index = np.random.choice(np.arange(len(neg_datas)))
        neg_data = neg_datas[neg_index]

        return [pos_data1, pos_data2, neg_data], [label, label, neg_label]

    def __len__(self):
        return self.y_clcs


train_dataset = SpectrumDataset(X_train, y_train, train_dict)
val_dataset = SpectrumDataset(X_test, y_test, test_dict)

if __name__=="__main__":
    print(data.keys())
    print(data['subclass_raw_dict'])
    print(len(data['subclass_raw_dict']))
    sdataset = SpectrumDataset(X_data, y_data, data_dict, y_clcs)
    sdataloader = iter(DataLoader(sdataset, batch_size=2))