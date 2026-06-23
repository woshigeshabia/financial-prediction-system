import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = yf.download(
    "BTC-USD",
    start="2020-01-01",
    end="2025-01-01"
)

data.to_csv("btc.csv")


df = pd.read_csv("btc.csv")

close = df['Close'].values.reshape(-1,1)

scaler = MinMaxScaler()
close = scaler.fit_transform(close)

window = 30

X = []
y = []

for i in range(window,len(close)):
    X.append(close[i-window:i])
    y.append(close[i])

X = np.array(X)
y = np.array(y)
model = LSTMModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(100):

    output = model(X_train)

    loss = criterion(output,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(epoch,loss.item())
