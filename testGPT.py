#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print("Available: ", tf.config.list_physical_devices())


# In[2]:


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


# In[3]:


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# In[4]:


chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# In[5]:


data = tf.convert_to_tensor(encode(text), dtype=tf.int64)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# In[6]:


import tensorflow as tf

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform(shape=(batch_size,), maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# In[17]:


@tf.function
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = tf.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = tf.reduce_mean(losses)
    model.train()
    return out


# In[8]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Head(layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.tril = tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)

        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = tf.matmul(q, k, transpose_b=True) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = tf.where(tf.equal(self.tril[:T,:T], 0), tf.fill(wei.shape,-float('inf')), wei) # (B,T,T)
        wei = tf.nn.softmax(wei) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = tf.matmul(wei,v) # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


# In[9]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = keras.Sequential([
            layers.Dense(4 * n_embd),
            layers.ReLU(),
            layers.Dense(n_embd),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.net(x)


# In[10]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class FeedForward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        self.net = keras.Sequential([
            layers.Dense(4 * n_embd),
            layers.ReLU(),
            layers.Dense(n_embd),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.net(x)


# In[11]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



# In[12]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GPTLanguageModel(keras.Model):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = keras.Sequential([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = layers.LayerNormalization() # final layer norm
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T)) # (T,C)
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
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(targets, logits))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits) # (B,C)
            # sample from the distribution
            idx_next = tf.random.categorical(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = tf.concat((idx,idx_next), axis=1) # (B,T+1)
        return idx


# In[19]:


import tensorflow as tf

model = GPTLanguageModel()
# print the number of parameters in the model

print(sum(tf.size(p) for p in model.trainable_variables)/1e6, 'M parameters')

# create a TensorFlow optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# generate from the model
context = tf.zeros((1, 1), dtype=tf.int32)
print(decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist()))


