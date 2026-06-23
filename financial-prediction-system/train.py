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