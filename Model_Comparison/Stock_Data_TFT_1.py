# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import pandas as pd
# !pip install yahoo_fin --upgrade
import yahoo_fin
import numpy as np
from yahoo_fin.stock_info import get_data
import yfinance as yf

import math, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
np.random.seed(42)
torch.manual_seed(42)

# %%
def load_data(stock, look_back):
    close_idx = stock.columns.get_loc('close')
    data_raw = stock.values
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)

    test_size = int(0.2 * len(data))
    x_train = data[:-test_size, :-1, :]
    y_train = data[:-test_size, -1, close_idx]

    x_test = data[-test_size:, :-1]
    y_test = data[-test_size:, -1, close_idx]

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]



# %%
class RNN(nn.Module):
    def __init__(self, nn_type, input_dim, hidden_dim, num_layers, output_dim,
                 cnn_channel=0, cnn_kernel_size=0, dropout=0, bidirectional=False):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nn_type = nn_type
        self.cnn_channel = cnn_channel
        if self.cnn_channel > 0:
            self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=self.cnn_channel,
                                 kernel_size=cnn_kernel_size, stride=1)
        self.rnn = getattr(nn, nn_type)(cnn_channel if self.cnn_channel > 0 else input_dim,
                                        hidden_dim, num_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=dropout)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim * num_layers * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        if self.cnn_channel > 0:
            x = self.cnn(x.swapaxes(1, 2)).swapaxes(1, 2)
        if self.nn_type == 'LSTM':
            out, (hidden, cell) = self.rnn(x)
        else:
            out, hidden = self.rnn(x)
        return self.dropout(self.fc(self.flatten(hidden.swapaxes(0, 1))))

# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout_p=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])

class Transformer(nn.Module):
    def __init__(self, feature_size=200, num_layers=2, dropout=0.1, sequence_length=60, output_size=1):
        super().__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, output_size)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(feature_size + 1, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        torch.nn.init.kaiming_uniform_(self.decoder.weight)

        self.fc1.bias.data.zero_()
        torch.nn.init.kaiming_uniform_(self.fc1.weight)

        self.fc2.bias.data.zero_()
        torch.nn.init.kaiming_uniform_(self.fc2.weight)

    def forward(self, src, prev_close):
        src = src.swapaxes(0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output[-1]

        output = torch.cat((output, prev_close.unsqueeze(1)), dim=1)

        output = self.relu(self.fc1(output))
        output = self.fc2(output)

        return output

    def _generate_square_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

# %%
def fetch_data_and_engineer_features(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    industry = stock.info['industry']
    sector = stock.info['sector']
    # print(f"Ticker Symbol: {stock_ticker}")
    # print(f"Industry: {industry}")
    # print(f"Sector: {sector}")

    # Fetch data
    stock_data = get_data(stock_ticker, start_date="01/01/2010", end_date="12/31/2019", index_as_date=True, interval="1d")
#log returns
    stock_data['Log_Return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))

    # 20 and 50 day moving day averages
    stock_data['SMA_20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['close'].rolling(window=50).mean()

    #exponential moving average/EMA
    stock_data['EMA_20'] = stock_data['close'].ewm(span=20, adjust=False).mean()

    #relative strength index/RSI
    window_length = 14
    delta = stock_data['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=window_length - 1, adjust=True, min_periods=window_length).mean()
    ema_down = down.ewm(com=window_length - 1, adjust=True, min_periods=window_length).mean()
    rs = ema_up / ema_down
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    #technical indicators
    stock_data['MACD'] = stock_data['close'].ewm(span=12).mean() - stock_data['close'].ewm(span=26).mean()
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9).mean()
    stock_data['Stochastic_%K'] = stock_data['close'].rolling(window=14).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min()), raw=True)
    stock_data['Stochastic_%D'] = stock_data['Stochastic_%K'].rolling(window=3).mean()

    #volume indicators
    stock_data['Volume_SMA_20'] = stock_data['volume'].rolling(window=20).mean()
    stock_data['Volume_Ratio'] = stock_data['volume'] / stock_data['Volume_SMA_20']

    #volatility indicators
    stock_data['Daily_Return_Volatility'] = stock_data['Log_Return'].rolling(window=14).std() * np.sqrt(252)
    stock_data['Garman_Klass_Volatility'] = np.sqrt(252 * 0.5 * ((np.log(stock_data['close'] / stock_data['close'].shift(1)))**2).rolling(window=14).mean())

    #momentum and trend indicators
    stock_data['Price_Momentum'] = stock_data['close'].pct_change(periods=14)
    stock_data['Rate_of_Change'] = stock_data['close'].pct_change(periods=9)
    stock_data['Trend'] = np.where(stock_data['close'] > stock_data['close'].shift(1), 1, -1)

    stock_data['Day_of_Week'] = stock_data.index.dayofweek
    stock_data['Month'] = stock_data.index.month

    # information ratio
    stock_data['info_ratio'] = stock_data['close'] / stock_data['close'].shift(14) / stock_data['Daily_Return_Volatility']

    stock_data['close_change'] = stock_data['close'] - stock_data['close'].shift(1)
    stock_data['range'] = stock_data['high'] - stock_data['low']
    stock_data['ATR'] = stock_data['range'].rolling(window=14).mean()
    stock_data['stoc_oscillator'] = (stock_data['close'] - stock_data['low'].rolling(window=14).min()) / (stock_data['high'].rolling(window=14).max() - stock_data['low'].rolling(window=14).min())

    stock_data['cum_ret'] = stock_data['close'] / stock_data.iloc[-1]['open']

    stock_data = stock_data.sort_index(ascending=False).dropna().iloc[::-1]
    # Cell with the first scaling, normalize to -1 and 1 - method used in LSTM
    return stock_data


# %%
input_size = 30
hidden_size = 128 #could increase to 256 but VRAM will be limiter
output_size = 1
lstm_layers = 6
dropout = 0.1
num_epochs = 150
device = 'cuda'
lr = 1e-4

# %%
def train(model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prev_close = data[:, -1, -1]
        optimizer.zero_grad()
        output = model(data, prev_close)
        loss = loss_function(output.squeeze(), target)
        train_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()
        # if (batch_idx + 1) % (len(train_loader) // 4) == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)], Batch Loss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()
        #     ))
    return train_loss / len(train_loader)

# %%
def test(model, device, test_loader, epoch, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            prev_close = data[:, -1, -1]
            output = model(data, prev_close)
            test_loss += loss_function(output.squeeze(), target).item()
    # print('Test epoch: {}, Test batch loss: {:.4f}'.format(
    #     epoch, test_loss / len(test_loader)
    # ))
    return test_loss / len(test_loader)

# %%
s_p = pd.read_csv(r'./constituents.csv')
full_ticker = s_p['Symbol'].tolist()
print(full_ticker)
exclude_list = ["ABNB", "BRK.B", "BF.B", "CARR", "CEG", "GEHC", "GEV", "KVUE", "OTIS", "SOLV", "VLTO"]
filtered_ticker = [ticker for ticker in full_ticker if ticker not in exclude_list]
test_tickers = filtered_ticker
test_tickers = filtered_ticker[:250]
# test_tickers = ['AMZN']

# %%
train_rmse_list = []
test_rmse_list = []
start_time = time.time()

stock_datas = {}
for stock_ticker in test_tickers:
    stock_datas[stock_ticker] = fetch_data_and_engineer_features(stock_ticker).drop(columns=['ticker'])
stock_data_full = pd.concat([value for _, value in stock_datas.items()])
data_scaler = MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
stock_data_temp = data_scaler.fit(stock_data_full)
output_scaler = MinMaxScaler(feature_range=(-1, 1)).set_output(transform="pandas")
output_scaler.fit(stock_data_full[['close']])

for stock_ticker in test_tickers:
    stock_data = stock_datas[stock_ticker]
    stock_data = data_scaler.transform(stock_data)

    look_back = 61
    x_train, y_train, x_test, y_test = load_data(stock_data, look_back)
    # print(y_train.size(), x_train.size())
    
    batch_size = 32
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size, shuffle=True)


    model = Transformer(
        feature_size=input_size,
        num_layers=lstm_layers,
        dropout=dropout,
        sequence_length=look_back - 1,
        output_size=output_size
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # print(model)
    # print(len(list(model.parameters())))
    # for i in range(len(list(model.parameters()))):
    #     print(list(model.parameters())[i].size())

    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs + 1):
        train_losses.append(train(model, device, train_loader, optimizer, epoch, loss_fn))
        val_losses.append(test(model, device, test_loader, epoch, loss_fn))
        if (val_losses[-1] == min(val_losses)):
            torch.save(model.state_dict(), 'model_weights_TFT.pth')
    # plt.plot(np.arange(1, num_epochs + 1), train_losses, label=r'train')
    # plt.plot(np.arange(1, num_epochs + 1), val_losses, label=r'val')
    # plt.legend()


    model.load_state_dict(torch.load('model_weights_TFT.pth'))
    prev_close_train = x_train[:, -1, -1]  # Extract the previous day's closing price for training data
    y_train_pred = model(x_train.to(device), prev_close_train.to(device)).to('cpu')
    print(np.shape(y_train_pred))

    prev_close_test = x_test[:, -1, -1]  # Extract the previous day's closing price for testing data
    y_test_pred = model(x_test.to(device), prev_close_test.to(device)).to('cpu')

    y_train_pred = output_scaler.inverse_transform(y_train_pred.detach().numpy().reshape(-1, 1))[:, 0]
    y_train = output_scaler.inverse_transform(y_train.detach().numpy().reshape(-1, 1))[:, 0]
    y_test_pred = output_scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1))[:, 0]
    y_test = output_scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))[:, 0]

    trainScore = math.sqrt(mean_squared_error(y_train, y_train_pred))
    # print('Train Score: %.2f RMSE' % (trainScore))
    train_rmse_list.append(trainScore)
    testScore = math.sqrt(mean_squared_error(y_test, y_test_pred))
    test_rmse_list.append(testScore)
    # print('Test Score: %.2f RMSE' % (testScore))

np.savetxt("TFT_train_1.csv", train_rmse_list, delimiter=",")
np.savetxt("TFT_test_1.csv", test_rmse_list, delimiter=",")
# np.loadtxt("file", delimiter=",")
print(time.time() - start_time)

# %%
plt.hist(test_rmse_list)
plt.savefig("TFT_1.png")
plt.show()
# figure, axes = plt.subplots(figsize=(15, 6))
# axes.xaxis_date()
# axes.plot(stock_data[len(stock_data)-len(y_test):].index, y_test, color = 'red', label = 'real')
# axes.plot(stock_data[len(stock_data)-len(y_test):].index, y_test_pred, color = 'blue', label = 'pred')
# plt.show()


