import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN_BILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(CNN_BILSTM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=100,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2 * 100, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, input_dim, seq_len]
        print(x.shape)
        x = self.conv(x)  # -> [batch, 128, reduced_seq_len]
        x = x.permute(0, 2, 1)  # for LSTM: [batch, seq_len, 128]
        x,state = self.lstm(x)   # -> [batch, seq_len, 2*hidden_dim]
        x = x.permute(0, 2, 1)  # for pooling: [batch, 2*hidden_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # -> [batch, 2*hidden_dim]
        x = self.fc(x)
        return self.sigmoid(x)


def train_model(model,train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

if __name__ == "__main__":
    # Example usage
    input_dim = 100
    hidden_dim = 100
    output_dim = 1
    model = CNN_BILSTM(input_dim, hidden_dim, output_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print the model architecture
    summary(model, input_size=(100,1000))
    # Create a random input tensor with shape (batch_size, seq_len, input_dim)
    # x = torch.randn(32, 1000, input_dim)
    # # train_model(model, x, None, None, None)  # Dummy train_model call for testing
    # # Forward pass
    # output = model(x)
    # print(output.shape)  # Should be (32, 1)



        
        
        