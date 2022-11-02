import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import LSTM
from torch import optim
from sklearn.model_selection import train_test_split
from ast import literal_eval
from sklearn import preprocessing
from scipy.fftpack import dct
from scipy.fftpack import idct

class MyDataset(Dataset):

    def __init__(self, datax, datay):
        super(Dataset).__init__()


        self.src = datax
        # self.src.astype(np.float32)

        self.label = datay
        # self.label.astype(np.float32)

    def __getitem__(self, index):
        return self.src[index], self.label[index]

    def __len__(self):
        return len(self.src)

def train_loop(dataloader, model, loss_fn, optimizer, which_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        if which_model == 1:
            X = X.squeeze(1)
        elif which_model == 2:
            X = X.unsqueeze(-1)
        else:
            pass
        # Compute prediction and loss
        # print(X.shape)
        X = X.to(device)
        y = y.to(device)
        pred = model(X.float())
        # print(pred.shape)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, which_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if which_model == 1:
                X = X.squeeze(1)
            elif which_model == 2:
                X = X.unsqueeze(-1)
            else:
                pass
            X = X.to(device)
            y = y.to(device)
            pred = model(X.float())
            # print(pred)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")

def trian_lstm(learning_rate, num_epoches, train_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 之前lstm模型保存的地址
    model_old = torch.load("/content/drive/MyDrive/有监督的多分类模型/目前的lstm模型/Lstm_all.pth")
    model = LSTM(1, 128, 4, 7, True).to(device)
    model.load_state_dict(model_old)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 2)
        print("Done!")

    # 新的lstm模型保存的地址
    torch.save(model.state_dict(), "/content/drive/MyDrive/有监督的多分类模型/目前的lstm模型/Lstm_all_1.pth")
    print("Saved PyTorch Model State to al Lstm_all_1.pth")

def changedata(data):
    newdata_t = []
    min_max_scaler = preprocessing.MinMaxScaler()
    for i in range(len(data)):
        tem = literal_eval(data.iloc[i, 0])  # 表示第i行，第1列（SCORE列）
        # tem=dct(tem, n=200)#dct变换
        tem = dct(tem)  # dct变换
        tem = tem[:5]
        tem = tem / 2
        tem = idct(tem, n=100)
        tem = tem / 100
        #     tem=standalize2(tem)#进行softmax标准化
        tem = min_max_scaler.fit_transform(tem.reshape(-1, 1))
        newdata_t.append(tem.reshape(1, -1)[0])
    #     newdata_t.append(tem)
    return newdata_t

def main():
    # data = pd.read_csv("data/all/1/need_fit2.csv")

    #数据的地址
    datax = np.load("/content/drive/MyDrive/有监督的多分类模型/al_x_01.npy")
    datay = np.load("/content/drive/MyDrive/有监督的多分类模型/al_y_01.npy")

    # print(datax)
    # print("---------------------")
    # print(datay)


    data_train = MyDataset(datax, datay)

    # 定义超参数
    batch_size = 50
    learning_rate = 2e-4 #学习率要慢慢调，不可太大
    num_epoches = 1

    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)


    trian_lstm(learning_rate, num_epoches, data_loader_train)


    # for i_batch, data_batch in enumerate(data_loader_train):
    #     print("-------------------", i_batch, '-------------------------')
    #     print(data_batch[0].shape)
    #     print(data_batch[1].shape)


if __name__ == "__main__":
    main()