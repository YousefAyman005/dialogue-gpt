import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. ENVIRONMENT SETUP ---
def _setup_tpu_env():
    # Critical for v5e stability
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1" # Use BFloat16 (Native to TPU)
    
    # Remove conflicting variables
    for key in ["TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"]:
        if key in os.environ:
            os.environ.pop(key)

_setup_tpu_env()

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# --- 2. HYPERPARAMETERS ---
batch_size = 32      # Per core
block_size = 256
max_iters = 5000
learning_rate = 3e-4
eval_interval = 200  # Check validation every 200 steps
eval_iters = 20      # Look at 20 batches to estimate loss
n_embd = 384
n_head = 6
n_layers = 6
dropout = 0.2

# Global Data (CPU)
_train_data = None
_val_data = None
_vocab_size = None

# --- 3. DATA SETUP ---
def _load_data():
    global _train_data, _val_data, _vocab_size
    import tiktoken
    
    path = 'data/movies/input.txt'
    if not os.path.exists(path):
        print("⚠️ No data file found. Using dummy data.")
        text = "Hello world " * 10000
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    _vocab_size = enc.n_vocab
    
    # Load to CPU tensor
    full_data = torch.tensor(enc.encode(text, allowed_special={'<|endoftext|>'}), dtype=torch.long)
    
    # Split Train/Val
    n = int(0.9 * len(full_data))
    _train_data = full_data[:n]
    _val_data = full_data[n:]
    
    return _vocab_size

# --- 4. BATCH GETTER (CPU Slicing) ---
def get_batch(split, device):
    # Select the correct data tensor (Global CPU tensors)
    data = _train_data if split == 'train' else _val_data
    
    # Slice on CPU
    ix = torch.randint(len(data) - block_size, (batch_size,), device='cpu')
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to TPU
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y

# --- 5. ESTIMATE LOSS (The TPU-Safe Version) ---
@torch.no_grad()
def estimate_loss(model, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss
        
        # 1. Calculate the mean loss for THIS core
        local_mean = losses.mean()
        
        # 2. SYNC: Average the loss across ALL 8 cores
        # This prevents crashes by forcing all cores to wait here
        global_mean = xm.all_reduce(xm.REDUCE_SUM, local_mean, scale=1.0/8)
        
        out[split] = global_mean.item()
        
    model.train()
    return out

# --- 6. MODEL (NanoGPT) ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_final(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

# --- 7. TRAINING WORKER ---
def _train_worker(index):
    try:
        device = torch_xla.device()
        rank = index
        print(f"✅ Core {rank} active on {device}")
        
        torch.manual_seed(1337 + rank)
        
        # Init Model
        model = GPT(_vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Hard sync before loop
        xm.rendezvous('init_complete')
        
        if rank == 0:
            print("🚀 Graph compiled. Starting loop...")
            start_time = time.time()

        model.train()
        
        for iter in range(max_iters):
            
            # --- VALIDATION BLOCK ---
            if iter % eval_interval == 0 and iter > 0:
                # estimate_loss contains an implicit sync (xm.all_reduce)
                # so it is safe to call here.
                losses = estimate_loss(model, device)
                
                if rank == 0:
                    dt = time.time() - start_time
                    print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {dt:.2f}s")
                    start_time = time.time() # Reset timer
                
                xm.save(model.state_dict(), 'checkpoint.pth')
                
                if rank == 0:
                    print(f"✅ Saved checkpoint to checkpoint.pth")
            # --- TRAINING BLOCK ---
            xb, yb = get_batch('train', device)
            
            # Forward/Backward
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Update (Implicit sync)
            xm.optimizer_step(optimizer)

    except Exception as e:
        print(f"❌ ERROR on Core {index}: {e}")
        raise e

# --- 8. MAIN LAUNCHER ---
if __name__ == '__main__':
    print("⏳ Loading data...")
    _vocab_size = _load_data()
    print(f"Data loaded. Vocab size: {_vocab_size}")
    
    print("🔥 Spawning 8 TPU processes (BF16 Mode)...")
    xmp.spawn(_train_worker, args=(), nprocs=None, start_method='fork')