# This is an example of how to implement an LSTM
# Code from the course DNN

import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd

# Download data

# gdd.download_file_from_google_drive('13DLnvpRhDO-D5JHCuPq85k_v-yMiMNCQ', '../../../../../Documents/sdu.extensions.2023.1.0.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/files/lingsmap_public-bare.zip', unzip=True)
# !gdown 'https://drive.google.com/uc?id=13DLnvpRhDO-D5JHCuPq85k_v-yMiMNCQ'
# !unzip '/content/lingsmap_public-bare.zip'
# assert Path('/content/lingsmap_public-bare').exists()

# Load data

filepaths = glob.glob('../../../../../Documents/sdu.extensions.2023.1.0.hotfix/exts/omni.sdu.ai/omni/sdu/ai/examples/files/lingsmap_public-bare/*/*.txt')
emails = []
emails_ascii = []
targets = []

for i, filepath in enumerate(tqdm(filepaths)):
    with open(filepath) as file:
        email = file.read()
    emails.append(email)
    emails_ascii.append([ord(c) for c in email])  # convert character to ascii code
    targets.append('spmsg' in filepath)

plt.hist([len(e) for e in emails], np.logspace(1, 5))
plt.xlabel('email text length')
plt.ylabel('#')
plt.xscale('log')
plt.show()

print(f'{len(emails)} emails')
print(f'percentage spam: {np.mean(targets):.2f}')
emails_ascii_flat = np.concatenate(emails_ascii)
print(f'ascii codes between {emails_ascii_flat.min()} and {emails_ascii_flat.max()}')

# lets read few examples our loaded data
for i in range(4):
    print(f'----  Spam: {targets[i]}  ----')
    print(emails[i], '\n')
    

# Task: Create and train a recurrent model to classify whether the emails are spam or not

class SpamDataset:
  
    def __init__(self, emails_ascii, targets, text_length, sample=False):
        self.emails_ascii = emails_ascii
        self.targets = targets
        self.text_length = text_length
        self.sample = sample

    def __len__(self):
        return len(self.emails_ascii)

    def __getitem__(self, i):
        text = self.emails_ascii[i]
        if self.sample:
            start_idx = np.random.randint(max(1, len(text) - self.text_length))
        else:
            start_idx = 0
        text = text[start_idx:start_idx+self.text_length]
        end_idx = len(text) - 1
        text = text + [0] * (self.text_length - len(text))
        assert len(text) == self.text_length
        text = np.array(text, dtype=int)
        return text, end_idx, self.targets[i]
    
n_valid = 250
data_train = SpamDataset(emails_ascii[:-n_valid], targets[:-n_valid], 100, sample=True)
data_valid = SpamDataset(emails_ascii[-n_valid:], targets[-n_valid:], 500)

print(f'{np.sum(targets[-n_valid:])} spam mails in the validation set')

loader_kwargs = dict(batch_size=32, num_workers=4)
loader_train = torch.utils.data.DataLoader(data_train, shuffle=True, **loader_kwargs)
loader_valid = torch.utils.data.DataLoader(data_valid, **loader_kwargs)


class SpamClassifier(nn.Module):

    def __init__(self, d=512, n_lstm_layers=1, dropout=0.):
        super().__init__()
        self.emb = nn.Embedding(128, d)  # max(emails_ascii) < 128
        self.lstm = nn.LSTM(d, d, n_lstm_layers, batch_first=True, dropout=dropout)
        self.lin = nn.Linear(d, 1)

    def forward(self, x, end_idx):  # (B, nx)
        x = self.emb(x)  # (B, nx, d)
        y = self.lstm(x)[0][torch.arange(len(x)), end_idx]  # (B, d)
        return self.lin(y).view(-1)  # (B,)
    
device = torch.device('cuda')

model = SpamClassifier().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, (30,))

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
lrs = []

pbar = tqdm(range(40))
for epoch in pbar:
    # train
    model.train()
    losses = []
    correct = total = 0
    for x, end_idx, y in loader_train:
        x, end_idx, y = x.to(device), end_idx.to(device), y.to(device)
        logits = model(x, end_idx)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        correct += ((torch.sigmoid(logits) > 0.5) == y).sum().item()
        total += len(x)
    train_loss = np.mean(losses)
    train_acc = correct / total

    # valid
    model.eval()
    losses = []
    correct = total = 0
    for x, end_idx, y in loader_valid:
        x, end_idx, y = x.to(device), end_idx.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x, end_idx)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        losses.append(loss.item())
        correct += ((torch.sigmoid(logits) > 0.5) == y).sum().item()
        total += len(x)
    valid_loss = np.mean(losses)
    valid_acc = correct / total

    # sched
    sched.step()
    
    # history
    lrs.append(next(iter(opt.param_groups))['lr'])

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    pbar.set_description(f'loss: {train_loss:.3f}/{valid_loss:.3f}, acc: {train_acc:.2f}/{valid_acc:.2f}')

# plot history
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].plot(train_losses, label='train')
axs[0].plot(valid_losses, label='valid')
axs[0].set_ylabel('loss')
axs[0].legend()
axs[1].plot(train_accuracies, label='train')
axs[1].plot(valid_accuracies, label='valid')
axs[1].set_ylabel('acc')
axs[1].set_ylim(0.8, 1)
axs[1].legend()
axs[2].plot(lrs)
axs[2].set_ylabel('lr')
axs[2].set_yscale('log')
plt.tight_layout()
plt.show()