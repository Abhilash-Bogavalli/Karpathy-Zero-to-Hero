import torch
import torch.nn as nn 
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#------
torch.manual_seed(1337)

with open("/Users/abhilashbogavalli/Desktop/AI/AI/Andrej/Karpathy-Zero-to-Hero/GPT/input.txt",'r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
num_of_unique_chars = len(chars)
vocab_size = num_of_unique_chars

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda c: "".join([itos[s] for s in c])


data = torch.tensor(encode(text),dtype =  torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = {"train":train_data,
     "val":val_data}[split]
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets = None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        
        else:
            B,T,C = logits.shape
            logitst = logits.view(B*T,C) # Reshaping the different batches to be of one and making sure C (num of classes) is the second shape of logits
            targets = targets.view(B*T) # Since the targets should match the logits first shape 
            loss = F.cross_entropy(logitst,targets) 
        
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx has shape B,T (say 4, 8)
        for _ in range(max_new_tokens):
            logits,loss = self(idx) # (B,T,C) 

            logits = logits[:,-1,:] # Taking the logits of the last character in every batch so B,C 

            probs = F.softmax(logits,dim =1)

            idx_next = torch.multinomial(probs,num_samples=1) # B,1 - it predicts what the next character is for every batch 

            idx = torch.cat((idx,idx_next),dim=1) # add the next coming element to our new idx which would be B,T+1 
        return idx 

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr = learning_rate)

for iter in range(max_iters):
    
    if iter % eval_interval == 0:

        losses = estimate_loss()

        print(f"Step {iter}, training loss {losses['train']:.4f}, validation loss {losses['val']:.4f} ")

    xb,yb = get_batch('train')

    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1),dtype = torch.long, device = device)
print(decode(m.generate(context,max_new_tokens = 500)[0].tolist()))
