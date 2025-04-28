import numpy as np
import time
import gc 
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,Dataset
from sklearn.metrics import classification_report
from torch.cuda import memory_summary
from util import torchmetrics_classification_report

import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import torchmetrics



class CNN_BILSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(CNN_BILSTM, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.Relu(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.Relu(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.Relu(),
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

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(2 * 100, 50)
        self.fc2 = nn.Linear(50, output_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, input_dim, seq_len]
        x = x.permute(0, 2, 1)  
        x = self.conv(x)  # -> [batch, 128, reduced_seq_len]
        x = x.permute(0, 2, 1)  # for LSTM: [batch, seq_len, 128]
        x,state = self.lstm(x)   # -> [batch, seq_len, 2*hidden_dim]
        x = x.permute(0, 2, 1)  # for pooling: [batch, 2*hidden_dim, seq_len]
        x = self.pool(x).squeeze(-1)  # -> [batch, 2*hidden_dim]
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


def train_model(model,train_loader,val_loader,criterion,optimizer,device,num_epochs=25):
    
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
            loss.backward()
            optimizer.step()
            
            del inputs, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()                   
            running_loss += (loss.item())
        # if epoch == 0:
            # print(len(train_loader.dataset))
            
        training_acc /= len(train_loader.dataset)
        train_accs.append(training_acc)
        avg_train_loss = running_loss / len(train_loader.dataset)
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
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        end = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | training acc: {training_acc:.4f} | val acc: {val_acc:.4f} | Time: {end - start:.2f}s")


    return train_losses, val_losses, train_accs, val_accs, val_outputs


if __name__ == "__main__":
    # Example usage
    input_dim = 150
    hidden_dim = 100
    output_dim = 1
    model = CNN_BILSTM(input_dim, hidden_dim, output_dim)    
    g = torch.Generator()
    g.manual_seed(42)
    # import torch

    # train_files_cbow = [os.path.join('../Data/cbow', f) for f in os.listdir('../Data/cbow') if 'train' in f]
    # train_files_skipgram = [os.path.join('../Data/skipgram', f) for f in os.listdir('../Data/skipgram') if 'train' in f]
    # train_files_glove = [os.path.join('../Data/glove', f) for f in os.listdir('../Data/glove') if 'train' in f]
    # # print(f"train_files_cbow: {train_files_cbow}")
    # train_files = [train_files_cbow, train_files_skipgram, train_files_glove]
    # # train_dataloader = CombinedFoldersDataset(*train_files,model_used='glove', batch_size=1000)
    
    # test_files_cbow = [os.path.join('../Data/cbow', f) for f in os.listdir('../Data/cbow') if 'test' in f]
    # test_files_skipgram = [os.path.join('../Data/skipgram', f) for f in os.listdir('../Data/skipgram') if 'test' in f]
    # test_files_glove = [os.path.join('../Data/glove', f) for f in os.listdir('../Data/glove') if 'test' in f]
    # test_files = [test_files_cbow, test_files_skipgram, test_files_glove]
    # print(test_files)
    # test_dataloader = CombinedFoldersDataset(*test_files,model_used='glove', batch_size=1000)
    
    # print(f"train_dataloader length: {len(train_dataloader)}")
    # exit()
    # glove_0 = np.load('../Data/glove/train0.npy')
    # glove_1 = np.load('../Data/glove/train1.npy')
    # glove_2 = np.load('../Data/glove/train2.npy')
    # glove_3 = np.load('../Data/glove/train3.npy')
    # glove = np.concatenate((glove_0, glove_1, glove_2,glove_3), axis=0)
    # print(f"glove shape: {glove.shape}")
    # del glove_0, glove_1, glove_2, glove_3
    # print(f"glove shape: {glove.shape}")
    cbow = np.load('../Data/cbow/train0.npy')
    glove = np.load('../Data/glove/train0.npy')
    train_data = np.concatenate((cbow, glove), axis=2)
    del cbow, glove
    print(f"train_data shape: {train_data.shape}")
    print("Loading data...")
    
    print(f"train_data shape: {train_data.shape}")
    glove_test = np.load('../Data/glove/test0.npy')
    cbow_test = np.load('../Data/cbow/test0.npy')
    test_data = np.concatenate((cbow_test, glove_test), axis=2)
    del cbow_test, glove_test
    
    train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(np.load('../Data/train_labels.npy')).float())
    test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(np.load('../Data/test_labels.npy')).float())
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    # Train the model
    train_losses, val_losses,train_accs,test_accs,val_ouputs = train_model(model,train_dataloader,test_dataloader,criterion,optimizer,device,num_epochs=num_epochs)    
    
    # train_losses = []
    # val_losses = []
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
    

    rep = classification_report(np.load('../Data/test_labels.npy'), np.array(val_ouputs))
    print(rep)

        
        
        