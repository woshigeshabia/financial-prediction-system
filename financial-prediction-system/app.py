from flask import Flask
from flask import render_template
from flask import request

import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from models.lstm import LSTMModel
from models.gru import GRUModel
from models.rnn import RNNModel

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

app = Flask(__name__)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    model_type = request.form["model"]

    df = pd.read_csv(file)

    close = df["Close"].values.reshape(-1,1)

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

    X = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    if model_type == "LSTM":

        model = LSTMModel()

        model.load_state_dict(
            torch.load(
                "weights/lstm.pth",
                map_location=DEVICE
            )
        )

    elif model_type == "GRU":

        model = GRUModel()

        model.load_state_dict(
            torch.load(
                "weights/gru.pth",
                map_location=DEVICE
            )
        )

    else:

        model = RNNModel()

        model.load_state_dict(
            torch.load(
                "weights/rnn.pth",
                map_location=DEVICE
            )
        )

    model.eval()

    with torch.no_grad():

        pred = model(X)

    pred = pred.cpu().numpy()

    pred = scaler.inverse_transform(pred)

    real = scaler.inverse_transform(y)

    plt.figure(figsize=(10,5))

    plt.plot(real,label="Real")

    plt.plot(pred,label="Predict")

    plt.legend()

    plt.savefig("static/result.png")

    return render_template(
        "result.html",
        model=model_type
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )