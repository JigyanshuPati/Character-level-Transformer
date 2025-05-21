import torch
from v2 import BigramLanguageModel, config, char_to_int, int_to_char, block_size, vocab_size, encode, decode, get_batch

config = {
    "block_size": 64,          # very short context window
    "batch_size": 64,          # smaller batch to fit easily in memory
    "n_embed": 64,             # very small embedding dimension
    "num_heads": 4,            # keep heads divisible, small attention
    "num_layers": 4,           # shallow network but enough to learn some patterns
    "dropout": 0.1,            # low dropout for small nets
    "learning_rate": 1e-3,     # keep learning rate high for fast learning
    "epochs": 10000,             # fewer epochs for quick runs
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
m = BigramLanguageModel(vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

def estimate_loss():
    m.eval()
    losses = {}
    for split in ['train', 'val']:
        loss_total = 0
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with torch.no_grad():
                _, loss = m(X, Y)
            loss_total += loss.item()
        losses[split] = loss_total / eval_iters
    m.train()
    return losses

for iter in range(epochs):
    if iter % 100 == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))

# Save model
torch.save({
    'model_state_dict': m.state_dict(),
    'config': config,
    'vocab': {
        'char_to_int': char_to_int,
        'int_to_char': int_to_char,
        'block_size': block_size,
        'vocab_size': vocab_size
    }
}, "harry_model.pt")

print("Model saved successfully!")