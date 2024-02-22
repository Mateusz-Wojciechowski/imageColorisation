from dataprepRGB import train_loader, test_loader
from AutoEncoderRGB import AutoEncoderRGB
import torch.optim as optim
from constants import LEARNING_RATE, NUM_EPOCHS
import torch.nn as nn
import torch

model = AutoEncoderRGB()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    model.train()

    for gray, color in train_loader:
        optimizer.zero_grad()
        output = model(gray)
        loss = loss_fn(output, color)
        loss.backward()
        optimizer.step()

        epoch_loss += loss

    epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}, Loss: {epoch_loss}")


torch.save(model.state_dict(), 'model75epochsNew.pth')