import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# ------------------ Hyperparameters ------------------
config = {
    "block_size": 64,          # very short context window
    "batch_size": 64,          # smaller batch to fit easily in memory
    "n_embed": 64,             # very small embedding dimension
    "num_heads": 4,            # keep heads divisible, small attention
    "num_layers": 4,           # shallow network but enough to learn some patterns
    "dropout": 0.1,            # low dropout for small nets
    "learning_rate": 1e-3,     # keep learning rate high for fast learning
    "epochs": 100,             # fewer epochs for quick runs
    "eval_iters": 20,          # quick evaluation loop
    "device": "mps" if torch.backends.mps.is_available() else "cpu"
}
# -----------------------------------------------------

block_size = config["block_size"]
batch_size = config["batch_size"]
n_embed = config["n_embed"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
dropout = config["dropout"]
learning_rate = config["learning_rate"]
epochs = config["epochs"]
eval_iters = config["eval_iters"]
device = config["device"]

# Load the dataset
with open("01 Harry Potter and the Sorcerers Stone.txt") as file:
    books = file.read()

# Preprocess the data
characters = sorted(list(set(books)))
vocab_size = len(characters)

# Create mapping from int to char and char to int
int_to_char = {i: c for i, c in enumerate(characters)}
char_to_int = {c: i for i, c in enumerate(characters)}

def encode(text):
    return [char_to_int[c] for c in text]

def decode(ints):
    return ''.join([int_to_char[i] for i in ints])

data = torch.tensor(encode(books), dtype=torch.long)

train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(head_size, num_heads)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads) for _ in range(num_layers)],
                                   nn.LayerNorm(n_embed))
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters=eval_iters):
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

# Utility for generating text from a prompt (after training)
def generate_from_prompt(model, prompt, max_new_tokens=100, temperature=1.0):
    idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    idx = model.generate(idx, max_new_tokens)
    return decode(idx[0].tolist())

# Model instantiation and training are handled in train_model.py
