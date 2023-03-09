import torch
import torch.nn as nn
from torch.nn import functional as F
import re

#Hyperparameters
batch_size = 32 #how many independent sequences to process at once
block_size = 8 #maximum context length for prediction
max_iters = 5000 #number of training iterations
lr = 1e-3 #learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_intervals = 500 
eval_iters = 200
n_embd = 32 #embedding dimension

#-------------------#

torch.manual_seed(1337) #for reproducibility

#text to train on
with open('Transcript.txt', 'r',encoding = 'utf-8') as f:
    text = f.read()

#unique characters in the text and the number of them
# chars = sorted(list(set(text)))
# chars = chars[:19]+ chars[46:48] + chars[74:]
splitted_text = re.findall(r'\S+', text)
chars = sorted(set(re.findall(r'\S+', text)))
vocab_size = len(chars)

#dictionary to convert characters to indices and vice versa
stoi = {ch:i for i,ch in enumerate(chars)} #mapping from caracter to index
itos = {i:ch for i,ch in enumerate(chars)} #mapping from index to caracter

encode = lambda s : [stoi[c] for c in s] #convert string to list of indices
decode = lambda l : ''.join([itos[i] for i in l]) #convert list of indices to string

#Train and test splits
data = torch.tensor(encode(splitted_text),dtype=torch.long)
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

@torch.no_grad() #never calling .bacward() on the loss, so we can save memory by not storing the gradients. IN this way we are more efficient
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

class Head(nn.Module):
    """ one head self-attention"""

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) #this isn't a variable but a paramter

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.key(x) # (B,T,C)
        #compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T) 
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim = -1) # (B,T,T)
        #note that we are normalizing bt dividing for C**0.5 to avoid one value to take an extreme value after softmax
        #perform the weighted aggregation of the values
        v = self.value(x) #(B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) = (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multi-head self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B,T,n_embd)
        out = self.proj(out) # (B,T,n_embd)
        return out
    
class FeedForward(nn.Module):
    """ a linear layer followed by a non linear layer"""

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self,x):
        return self.net(x)
    #the feed forward is done on a per token level, this is an individual thinking step done after the self-attention

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,n_embd, n_head):
        #n_embd: embedding dimension, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head #size of each head
        self.sa = MultiHeadAttention(n_head, head_size) #self-attention
        self.ffwd = FeedForward(n_embd) #feed forward

    def forward(self,x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x



#Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits of the next token from the embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #embedding layer for tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #embedding layer for position
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) #linear layer to map the output of the last block to the vocab size


    def forward(self, idx, targets = None):
        B, T = idx.shape
        

        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd) note that pos_emb is broadcasted to B,T,n_embd
        x = self.blocks(x) # (B,T,n_embd)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            #crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:] # (B,block_size)
            #get the preditctions
            logits, loss =  self(idx_cond)
            #we consider only the last time step
            logits = logits[:,-1,:]
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            #append the new index to the sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B,T+1)

        return idx
    

model = BigramLanguageModel().to(device)

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

