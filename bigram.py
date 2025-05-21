# Contents of /nanogpt/nanogpt/data.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
config = {
"device": "mps" if torch.backends.mps.is_available() else "cpu"
} 
device = config["device"]
# Load the dataset
with open("/Users/jigyanshupati/nanogpt/01 Harry Potter and the Sorcerers Stone.txt") as file:
    books = file.read()

# Preprocess the data
characters = sorted(list(set(books)))
vocab_size = len(characters)
print("Number of unique characters:", vocab_size)

# Create mapping from int to char and char to int
int_to_char = {i: c for i, c in enumerate(characters)}
char_to_int = {c: i for i, c in enumerate(characters)}

import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")
#function to convert text to integers
def encode(text):
    return enc.encode(text)
#function to convert integers to text
def decode(ints):
    return enc.decode(ints)

# Convert the data to tensor
data = torch.tensor(encode(books), dtype=torch.long)

# Split the data into training and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# Function to get a random batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Define the model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,C)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B*T,C)
            targets = targets.view(B * T)  # (B*T,)
            loss = F.cross_entropy(logits, targets)  # (B*T,)
            return logits, loss
        else:
            return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # (B, T)
            logits = self(idx_cond)  # get only logits here
            logits = logits[:, -1, :]  # focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def estimate_loss(model, get_batch, eval_iters=100):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
# Set parameters
block_size = 8
batch_size = 32
torch.manual_seed(42)

# Initialize the model
m = BigramLanguageModel(vocab_size, block_size)

# Training loop
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
epochs=1000
start_time = time.time()

for i in range(epochs):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        losses = estimate_loss(m, get_batch)
        print(f"Step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

# Generate text
print(decode(m.generate(xb[:1], max_new_tokens=500)[0].tolist()))