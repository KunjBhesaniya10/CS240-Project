import numpy as np
import time
import gc 
import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from util import *
import argparse
import numpy as np
import torch

class CNN_BILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(CNN_BILSTM, self).__init__()

        def multi_kernel_conv(in_channels, out_channels):
            return nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
                for k in [3, 5, 7]  # multiple kernel sizes
            ])

        self.conv1 = multi_kernel_conv(input_dim, 32)
        self.conv2 = multi_kernel_conv(32 * 3, 64)
        self.conv3 = multi_kernel_conv(64 * 3, 128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=128 * 3,
            hidden_size=100,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(2 * 100, 50)
        self.fc2 = nn.Linear(50, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]

        x = torch.cat([self.relu(conv(x)) for conv in self.conv1], dim=1)
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.cat([self.relu(conv(x)) for conv in self.conv2], dim=1)
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.cat([self.relu(conv(x)) for conv in self.conv3], dim=1)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x, state = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.adaptive_pool(x).squeeze(-1)

        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def train_model(model,train_loader,val_loader,criterion,optimizer,device,num_epochs=25,batch_size=64):
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    print(f"training on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        training_acc = 0.0
        start = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()  # outputs shape: [batch]
            training_acc += ((outputs > 0.5) == labels).float().sum().item()
            loss = criterion(outputs, labels.float())  # BCE requires float labels
            running_loss += (loss.item())
            loss.backward()
            optimizer.step()
            
            del inputs, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()                   
            
        training_acc /= len(train_loader.dataset)
        train_accs.append(training_acc)
        avg_train_loss = (running_loss*batch_size) / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

    
        # Validation loop   
        model.eval()
        val_running_loss = 0.0
        val_acc = 0.0
        val_outputs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                val_running_loss += loss.item()
                val_acc += ((outputs > 0.5) == labels).float().sum().item()
                val_outputs.extend((outputs > 0.5).cpu().numpy())
               
                del inputs, labels, outputs
                gc.collect()
                torch.cuda.empty_cache()
                        
        val_acc /= len(val_loader.dataset)
        val_accs.append(val_acc)
        avg_val_loss = val_running_loss*batch_size/ len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        end = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | training acc: {training_acc:.4f} | val acc: {val_acc:.4f} | Time: {end - start:.2f}s")


    return train_losses, val_losses, train_accs, val_accs, val_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',type=str, 
                        default='cbow-glove',
                        help='Model to use: cbow-glove, skipgram-glove, glove, skipgram-glove')
    parser.add_argument('--niters', type=int, default=10, help='Number of iterations for training')
    args = parser.parse_args()

    embeds = args.model.split('-')
    input_dict = {'cbow': 100, 'skipgram': 300, 'glove': 50}
   
    input_dim = 0
    for embed in embeds:
        if embed not in input_dict:
            raise ValueError(f"Invalid model name: {embed}. Choose from {list(input_dict.keys())}.")
        input_dim += input_dict[embed]
    hidden_dim = 100
    output_dim = 1

    model = CNN_BILSTM(input_dim, hidden_dim, output_dim)    
    g = torch.Generator()
    g.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train_files =[[os.path.join(f"../Data/{embed}", f) for f in os.listdir(f"../Data/{embed}") if 'train' in f] for embed in embeds]
    
    train_dataset = CombinedFoldersDataset(train_files,model_used=args.model, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    test_files =[[os.path.join(f"../Data/{embed}", f) for f in os.listdir(f"../Data/{embed}") if 'test' in f] for embed in embeds]    
    test_dataset = CombinedFoldersDataset(test_files,model_used=args.model, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train the model
    train_losses,val_losses,train_accs,test_accs,val_ouputs = train_model(model,
                                                                          train_dataloader,
                                                                          test_dataloader,
                                                                          criterion,
                                                                          optimizer,
                                                                          device,
                                                                          num_epochs=args.niters)   
    torch.save(model.state_dict(), f"{args.model}_model.pth")
    
    plot_save_name = args.model
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss vs Epochs - {plot_save_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_vs_epochs_{plot_save_name}.png')
    plt.show()
    plt.clf()
    
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epochs - {plot_save_name}')
    plt.legend()
    plt.savefig(f'accuracy_vs_epochs_{plot_save_name}.png')
    plt.show()
    
    test_labels = np.load('../Data/test_labels.npy')
    rep = classification_report(test_labels, np.array(val_ouputs),digits=4)
    conf = confusion_matrix(test_labels, np.array(val_ouputs))
    print(rep)
    print(conf)

        
        
        