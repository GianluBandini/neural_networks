import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
batch_size = 32 #how many independent sequences to process at once
block_size = 8 #maximum context length for prediction
max_iters = 3000 #number of training iterations
lr = 1e-3 #learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_intervals = 200 
eval_iters = 200

#-------------------#

torch.manual_seed(1337) #for reproducibility

#text to train on
with open('Transcript.txt', 'r',encoding = 'utf-8') as f:
    text = f.read()

#unique characters in the text and the number of them
chars = sorted(list(set(text)))
vocab_size = len(chars)

#dictionary to convert characters to indices and vice versa
stoi = {ch:i for i,ch in enumerate(chars)} #mapping from caracter to index
itos = {i:ch for i,ch in enumerate(chars)} #mapping from index to caracter

encode = lambda s : [stoi[c] for c in s] #convert string to list of indices
decode = lambda l : ''.join([itos[i] for i in l]) #convert list of indices to string

#Train and test splits
data = torch.tensor(encode(text),dtype=torch.long)
n = int(len(data)*0.9) #90% train, 10% test
train_data = data[:n] #first 90% of data
val_data = data[n:] #last 10% of data

#data loader
def get_batch(split):
    data = train_data if split=='train' else val_data #select train or test data
    ix = torch.randint(len(data)-block_size,(batch_size,)) #random starting indices
    x = torch.stack([data[i:i+block_size] for i in ix]) #input sequences
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #target sequences
    x, y = x.to(device), y.to(device)
    return x,y

@torch.no_grad() 
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



#Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):

        logits = self.token_emb(idx)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens):
        # idx is (B,T) tensor of indices
        for _ in range(max_tokens):
            logits =  self.token_emb(idx)
            #we consider only the last time step
            logits = logits[:,-1,:]
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            #append the new index to the sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B,T+1)

        return idx
    

model = BigramLanguageModel(vocab_size).to(device)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training loop
for iter in range(max_iters):
    
    #wvaluate the loss on train and test data
    if iter % eval_intervals == 0:
        losses = estimate_loss()
        print('step: {}, train loss: {:.4f}, test loss: {:.4f}'.format(iter, losses['train'], losses['val']))

    #sample a batch of data
    x,y = get_batch('train')

    #evaluate the loss
    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate text
context = torch.zeros(1,1).long().to(device)
print(decode(model.generate(context, max_tokens=500).squeeze().tolist()))

