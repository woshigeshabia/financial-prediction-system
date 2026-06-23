class RNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        out,_ = self.rnn(x)

        return self.fc(out[:,-1,:])