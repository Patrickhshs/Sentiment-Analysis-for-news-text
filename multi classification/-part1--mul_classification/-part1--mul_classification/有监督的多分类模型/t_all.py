import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from model import LSTM
from torch import optim
from sklearn.model_selection import train_test_split

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

        if batch % 100 == 0:
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

def trian_lstm(learning_rate, num_epoches, train_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTM(1, 128, 4, 7, True).to(device)

    loss_fn = nn.CrossEntropyLoss()#用于计算模型的输出和真实的y之间的差距的

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)#根据上面算出来的loss，来进行模型参数的学习和更新

    for t in range(num_epoches):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, 2)
        test_loop(test_loader, model, loss_fn, 2)
        print("Done!")

    torch.save(model.state_dict(), "data/all/Lstm_all.pth")
    print("Saved PyTorch Model State to Lstm_all.pth")

def main():
    #
    datax = np.load("data/all/all_x.npy")
    datay = np.load("data/all/all_y.npy")
    #数据切分成训练集和测试集
    train_x, eval_x, train_y, eval_y = train_test_split(datax, datay, test_size=0.17, random_state=2020)

    #构建Datset数据，用于模型的输入
    data_train = MyDataset(train_x, train_y)

    data_eval = MyDataset(eval_x, eval_y)

    # 定义超参数
    batch_size = 256
    learning_rate = 1e-4
    num_epoches = 30

    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_loader_eval = DataLoader(data_eval, batch_size=batch_size, shuffle=False)

    trian_lstm(learning_rate, num_epoches, data_loader_train, data_loader_eval)


    # for i_batch, data_batch in enumerate(data_loader_train):
    #     print("-------------------", i_batch, '-------------------------')
    #     print(data_batch[0].shape)
    #     print(data_batch[1].shape)


if __name__ == "__main__":
    main()