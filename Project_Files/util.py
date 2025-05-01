import torch
import torchmetrics
import time 
import numpy as np
from torch.utils.data import Dataset
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import classification_report

def torchmetrics_classification_report(preds, targets, num_classes):

    # Metrics without averaging (per class)
    precision = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average=None)
    recall = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average=None)
    f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average=None)

    # Macro average (overall)
    macro_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes, average="macro")
    macro_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes, average="macro")
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    # Compute all
    prec_vals = precision(preds, targets)
    rec_vals = recall(preds, targets)
    f1_vals = f1(preds, targets)
    acc_val = accuracy(preds, targets)

    macro_prec = macro_precision(preds, targets)
    macro_rec = macro_recall(preds, targets)
    macro_f1_val = macro_f1(preds, targets)

    # Print nicely
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("="*45)
    for i in range(num_classes):
        print(f"{i:<10} {prec_vals[i]:<10.4f} {rec_vals[i]:<10.4f} {f1_vals[i]:<10.4f}")
    print("="*45)
    print(f"{'Overall':<10} {macro_prec:<10.4f} {macro_rec:<10.4f} {macro_f1_val:<10.4f}")
    print(f"Accuracy: {acc_val:.4f}")

def get_npy_shape(path):
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
    return shape

class FolderDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.total_entries = 0
        self.curr_file_idx =0
        self.curr_file = np.load(file_paths[0])
        self.curr_file_size = self.curr_file.shape[0]
            
    def __len__(self):
        return (self.curr_file_size*len(self.file_paths))

    def __getitem__(self, idx):
        file_idx = idx // self.curr_file_size
        entry_idx = idx % self.curr_file_size
        if file_idx != self.curr_file_idx:
            self.curr_file_idx = file_idx
            del self.curr_file
            self.curr_file = np.load(self.file_paths[file_idx])
            self.curr_file_size = self.curr_file.shape[0]
        entry = self.curr_file[entry_idx]
        return torch.from_numpy(entry).float()
        

class CombinedFoldersDataset(Dataset):
    def __init__(self, folder_files:list, model_used='cbow-glove', is_train=True):
        self.folders = [FolderDataset(folder_files[i]) for i in range(len(folder_files))]
        self.model_used = model_used
        self.is_train = is_train
        if is_train:
            self.labels = np.load('../Data/train_labels.npy')
        else:   
            self.labels = np.load('../Data/test_labels.npy')
        
                
    def __len__(self):
        return self.folders[0].curr_file_size*len(self.folders[0].file_paths)

    def __getitem__(self, idx):

        batch_data = []
        for folder in self.folders:
            batch_data.append(folder[idx])    
        train_data = torch.cat(batch_data, dim=1)
        del batch_data
        
        labels = self.labels[idx]
        torch.cuda.empty_cache()
        
        return  train_data , labels
 


if __name__ == '__main__':
    # Example usage
    train_files_cbow = [os.path.join('../Data/cbow', f) for f in os.listdir('../Data/cbow') if 'train' in f]
    train_files_skipgram = [os.path.join('../Data/skipgram', f) for f in os.listdir('../Data/skipgram') if 'train' in f]
    train_files_glove = [os.path.join('../Data/glove', f) for f in os.listdir('../Data/glove') if 'train' in f]
    # print(f"train_files_cbow: {train_files_cbow}")
    train_files = [train_files_cbow, train_files_skipgram, train_files_glove]
    train_dataset = CombinedFoldersDataset(train_files,model_used='skipgram-glove', is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    
    
    test_files_cbow = [os.path.join('../Data/cbow', f) for f in os.listdir('../Data/cbow') if 'test' in f]
    test_files_skipgram = [os.path.join('../Data/skipgram', f) for f in os.listdir('../Data/skipgram') if 'test' in f]
    test_files_glove = [os.path.join('../Data/glove', f) for f in os.listdir('../Data/glove') if 'test' in f]
    test_files = [test_files_cbow, test_files_skipgram, test_files_glove]
    # print(test_files)
    
    test_dataset = CombinedFoldersDataset(test_files,model_used='skipgram-glove', is_train=False)