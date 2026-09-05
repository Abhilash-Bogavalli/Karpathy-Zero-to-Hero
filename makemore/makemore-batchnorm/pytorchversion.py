import torch
import torch.nn as nn
from torch.nn import functional as F
import random
context_length = 4
n_embd = 20
n_hidden = 700
batch_size = 32
num_epochs = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'
words = open('/Users/abhilashbogavalli/Desktop/AI/AI/Andrej/Karpathy-Zero-to-Hero/makemore/makemore-batchnorm/names.txt', 'r').read().splitlines()

chars = sorted(list(set("".join(words))))
vocab_size = len(chars) + 1
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

random.shuffle(words)

n = int(0.8*len(words))
train_words = words[:n]
val_data = words[n:]

@torch.no_grad()
def build_dataset(split):
    data = {"train":train_words,
            "val":val_data}[split]
    X,Y = [],[]

    for word in data:
        word = word + "."
        context = [0]*context_length
        for char in word:
            X.append(context)
            ix = stoi[char]
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class OurMlp(nn.Module):

    def __init__(self,vocab_size,n_embd,context_length,n_hidden):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size,n_embd)
        self.ly1 = nn.Linear(n_embd*context_length,n_hidden)
        self.batchlayer = nn.BatchNorm1d(n_hidden)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.ly2 = nn.Linear(n_hidden,vocab_size)
    
    def forward(self,x):
        x = self.embedding_table(x)
        x = x.view(-1,x.shape[-2]*x.shape[-1])
        batch_logits = self.ly1(x)
        normalized_guys = self.batchlayer(batch_logits)
        almost_logits = self.tanh(normalized_guys)
        almost_logits = self.dropout(almost_logits)
        logits = self.ly2(almost_logits)

        return logits 

Xtr,Ytr = build_dataset("train")

train_dataset = CharDataset(Xtr, Ytr)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = OurMlp(vocab_size, n_embd, context_length, n_hidden).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

@torch.no_grad()
def evaluate(loader):
    model.eval()                     # BatchNorm switches to running stats
    total_loss = 0
    num_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += loss.item()
        num_batches += 1
    model.train()                    # switch back so training continues normally after this
    return total_loss / num_batches

Xval, Yval = build_dataset("val")
val_dataset = CharDataset(Xval, Yval)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)   # no shuffle needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    train_loss = total_loss / num_batches
    val_loss = evaluate(val_loader)
    print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")
    
model.eval()
for _ in range(10):
    context = [0] * context_length
    our_word = []
    while True:
        x = torch.tensor([context])            
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        new_ix = torch.multinomial(probs, num_samples=1).item()
        if new_ix == 0:
            print("".join(our_word))
            break
        our_word.append(itos[new_ix])
        context = context[1:] + [new_ix]        



