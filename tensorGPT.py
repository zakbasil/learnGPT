#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


# In[2]:


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# In[3]:


# Train and test splits
data = tf.convert_to_tensor(encode(text), dtype=tf.int32)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# In[15]:


g1 = tf.random.Generator.from_seed(1331)

@tf.function
def stackTrainData(i):
    return tf.gather(train_data, tf.range(i,i+block_size))
@tf.function
def stackValData(i):
    return tf.gather(val_data, tf.range(i,i+block_size))

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = g1.uniform(minval=0,maxval=len(data)-block_size,shape=[batch_size,],dtype=tf.dtypes.int32)
    
    if(split == 'train'):
        x = tf.stack(tf.map_fn(stackTrainData, ix, dtype=tf.int32))
        y = tf.stack(tf.map_fn(stackTrainData, ix+1, dtype=tf.int32))
    elif(split == 'val'):
        x = tf.stack(tf.map_fn(stackValData, ix, dtype=tf.int32))
        y = tf.stack(tf.map_fn(stackValData, ix+1, dtype=tf.int32))
    print("getBatch",x.shape,y.shape)
    return x, y


# In[5]:


@tf.function
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = tf.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.numpy().tolist()
        out[split] = tf.math.reduce_mean(losses)
    return out


# In[6]:


class Head(layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.tril = tf.linalg.LinearOperatorLowerTriangular(tf.ones((block_size, block_size)))
        self.dropout = layers.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = tf.where(tf.equal(self.tril.to_dense()[None, :T, :T], 0), float('-inf'), wei) # (B, T, T)
        wei = tf.nn.softmax(wei,axis=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# In[7]:


class MultiHeadAttention(layers.Layer):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        out = tf.concat(head_outputs, axis=-1)
        out = self.dropout(out)
        out = self.proj(out)
        return out


# In[8]:


from tensorflow.keras import Sequential
class FeedFoward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super(FeedFoward,self).__init__()
        self.net = Sequential()
        self.net.add(layers.Dense(4 * n_embd))
        self.net.add(layers.ReLU())
        self.net.add(layers.Dense(n_embd))
        self.net.add(layers.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


# In[9]:


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super(Block,self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))


# In[10]:


class GPTLanguageModel(layers.Layer):

    def __init__(self):
        super(GPTLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = keras.Sequential([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = layers.LayerNormalization() # final layer norm
        self.lm_head = layers.Dense(vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self._init_weights()
        

    
    def _init_weights(self):
        for module in self.submodules:
            if isinstance(module, layers.Dense):
                module.kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
                if module.use_bias:
                    module.bias_initializer = keras.initializers.Zeros()
            elif isinstance(module, layers.Embedding):
                module.embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    


    def call(self, idx, targets=None):
        B, T = idx.shape
        print(B,T,idx.shape)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T, dtype=tf.int32)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B*T, C))
            targets = tf.reshape(targets, (B*T,))
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # sample from the distribution
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)  # (B, 1)
            # append sampled index to the running sequence
            idx = tf.concat([idx, idx_next], axis=1)  # (B, T+1)
        return idx


# In[11]:


model = GPTLanguageModel()


# In[16]:


# create a PyTorch optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


# In[ ]:





# In[ ]:





# In[ ]:




