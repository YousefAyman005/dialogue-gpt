import math
import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- STEP 1: ROBUST ENVIRONMENT SETUP ---
def _setup_tpu_env():
    """Forces Kaggle/Colab to see all 8 TPU cores."""
    if 'kaggle' in os.environ.get('KAGGLE_URL_BASE', '') or os.path.exists('/kaggle'):
        os.environ.setdefault('PJRT_DEVICE', 'TPU')
        # Remove the variables that cause the "1 vs 8 cores" crash
        for key in ["TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"]:
            if key in os.environ:
                os.environ.pop(key)

_setup_tpu_env()

# --- STEP 2: IMPORT XLA ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    _xla_available = True
except ImportError:
    _xla_available = False
    print("⚠️ WARNING: torch_xla not found. Running on CPU/GPU.")

# Helper to get env vars safely
def _env_int(name, default): return int(os.environ.get(name, default))
def _env_float(name, default): return float(os.environ.get(name, default))

# --- HYPERPARAMETERS ---
batch_size = _env_int("BATCH_SIZE", 64) # Increased for TPU efficiency
block_size = _env_int("BLOCK_SIZE", 256)
max_iters = _env_int("MAX_ITERS", 5000)
learning_rate = _env_float("LEARNING_RATE", 3e-4)
eval_interval = _env_int("EVAL_INTERVAL", 500)
n_embd = _env_int("N_EMBD", 384)
n_head = _env_int("N_HEAD", 6)
n_layers = _env_int("N_LAYERS", 6)
dropout = _env_float("DROPOUT", 0.2)
tpu_cores = 8 # Always 8 for v5e-8

# Global vars for data
_train_data = None
_val_data = None
_vocab_size = None

def _load_data():
    global _train_data, _val_data, _vocab_size
    import tiktoken
    
    # Ensure file exists
    path = 'data/movies/input.txt'
    if not os.path.exists(path):
        # Fallback for testing if file missing
        print(f"⚠️ File {path} not found! Creating dummy data.")
        text = "Hello world " * 10000
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    _vocab_size = enc.n_vocab
    data = torch.tensor(enc.encode(text, allowed_special={'<|endoftext|>'}), dtype=torch.long)
    
    n = int(0.9 * len(data))
    _train_data = data[:n]
    _val_data = data[n:]
    return _vocab_size

# --- MODEL COMPONENTS (Unchanged) ---
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
        # Scaled Dot-Product Attention
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
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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
        x = x + self.ffwd(self.ln2(x))
        return x

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
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# --- TRAINING UTILS ---
def get_batch(split, data_source):
    # data_source is already on TPU
    data = data_source[split]
    ix = torch.randint(len(data) - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, ctx_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval // 10, device=ctx_data['train'].device)
        for k in range(len(losses)):
            X, Y = get_batch(split, ctx_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- WORKER FUNCTION ---
def _train_worker(index):
    # 1. Setup Device
    # 'index' IS the rank (0-7), so we don't need xm.get_ordinal()
    rank = index 
    
    # Fix the deprecation warning (use xm.device instead of xla_device)
    device = xm.device() 
    
    print(f"Core {rank} active on {device}")
    
    # 2. Sync Seeds for reproducibility
    torch.manual_seed(1337 + rank)

    # 3. Move Data to TPU
    ctx_data = {
        'train': _train_data.to(device),
        'val': _val_data.to(device)
    }
    
    # 4. Model & Optimizer
    model = GPT(_vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 5. Training Loop
    model.train()
    for iter in range(max_iters):
        xb, yb = get_batch('train', ctx_data)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # XLA Step: Syncs gradients and executes the graph
        xm.optimizer_step(optimizer)

        if iter % eval_interval == 0:
            losses = estimate_loss(model, ctx_data)
            
            # ONLY Rank 0 (Master) prints and saves
            if rank == 0:
                print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                xm.save(model.state_dict(), 'checkpoint.pth')
# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Load data once in the main process
    _vocab_size = _load_data()
    print(f"Data loaded. Vocab size: {_vocab_size}")
    
    if _xla_available:
        print("🚀 Launching on TPU v5e-8...")
        xmp.spawn(_train_worker, args=(), nprocs=None, start_method='fork') # ✅ Correct
    else:
        print("❌ TPU not found. Check environment.")