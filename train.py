import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Setup TPU environment for Kaggle BEFORE importing torch_xla
def _setup_tpu_env():
    """Configure environment variables for Kaggle TPU."""
    if os.path.exists('/kaggle'):
        os.environ.setdefault('PJRT_DEVICE', 'TPU')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_setup_tpu_env()

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    _xla_available = True
except ImportError:
    torch_xla = None
    xm = None
    xmp = None
    _xla_available = False

def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default

def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default

# hyperparameters
batch_size = _env_int("BATCH_SIZE", 32)
block_size = _env_int("BLOCK_SIZE", 256)
max_iters = _env_int("MAX_ITERS", 20000)
eval_interval = _env_int("EVAL_INTERVAL", 1000)
learning_rate = _env_float("LEARNING_RATE", 3e-4)
min_lr = _env_float("MIN_LR", 3e-5)
warmup_iters = _env_int("WARMUP_ITERS", 500)
lr_decay_iters = _env_int("LR_DECAY_ITERS", max_iters)
weight_decay = _env_float("WEIGHT_DECAY", 0.1)
eval_iters = _env_int("EVAL_ITERS", 50)
n_embd = _env_int("N_EMBD", 768)
n_head = _env_int("N_HEAD", 12)
n_layers = _env_int("N_LAYERS", 9)
ffn_mult = _env_int("FFN_MULT", 8)
dropout = _env_float("DROPOUT", 0.3)
tpu_cores = _env_int("TPU_NUM_CORES", 8)
seed = _env_int("SEED", 1337)

if n_embd % n_head != 0:
    raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

def _should_use_xla():
    if not _xla_available:
        return False
    if os.environ.get("USE_TPU") == "1":
        return True
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if os.path.exists('/kaggle') and os.path.exists('/dev/accel0'):
        return True
    return any(os.environ.get(name) for name in ("COLAB_TPU_ADDR", "TPU_NAME", "XRT_TPU_CONFIG"))

use_xla = _should_use_xla()
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _log_initial_device():
    if use_xla:
        pjrt = os.environ.get("PJRT_DEVICE", "").upper() or "unset"
        tpu_cores_env = os.environ.get("TPU_NUM_CORES", "unset")
        print(f"Initial device selection: XLA/TPU (PJRT_DEVICE={pjrt}, TPU_NUM_CORES={tpu_cores_env})")
    else:
        print(f"Initial device selection: {default_device}")

# Global variables for data - will be initialized in main process only
_train_data = None
_val_data = None
_vocab_size = None
_enc = None

def _load_data():
    """Load and prepare the dataset. Called once in main process."""
    global _train_data, _val_data, _vocab_size, _enc
    
    import tiktoken
    
    with open('data/movies/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    _enc = tiktoken.get_encoding("gpt2")
    _vocab_size = _enc.n_vocab
    
    encode = lambda s: _enc.encode(s, allowed_special={'<|endoftext|>'})
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    _train_data = data[:n]
    _val_data = data[n:]
    
    return _vocab_size

def get_batch(split, device, train_data, val_data):
    data = train_data if split == 'train' else val_data
    max_start = len(data) - block_size
    if max_start <= 0:
        raise ValueError("block_size must be smaller than the dataset length.")
    data_device = data.device
    ix = torch.randint(0, max_start, (batch_size,), device=data_device)
    offsets = torch.arange(block_size, device=data_device)
    x = data[ix[:, None] + offsets]
    y = data[ix[:, None] + offsets + 1]
    if data_device != device:
        x = x.to(device)
        y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, device, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, device, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

class Head(nn.Module):
    def __init__(self, head_size, vocab_size):
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
        wei = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, vocab_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, vocab_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd, ffn_mult):
        super().__init__()
        hidden_dim = ffn_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class blocks(nn.Module):
    def __init__(self, n_embd, n_head, vocab_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, vocab_size)
        self.ffwd = FeedFoward(n_embd, ffn_mult)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[blocks(n_embd, n_head=n_head, vocab_size=vocab_size) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def build_model(vocab_size):
    return BigramLanguageModel(vocab_size)

def _is_master():
    return (not use_xla) or xm.is_master_ordinal()

def _save_checkpoint(checkpoint, path):
    if use_xla:
        xm.save(checkpoint, path)
    else:
        torch.save(checkpoint, path)

def _train_worker(index):
    """Training worker function for TPU multiprocessing."""
    global _train_data, _val_data, _vocab_size
    
    if use_xla:
        device = xm.xla_device()
        print_fn = xm.master_print
        world_size = xm.xrt_world_size()
        rank = xm.get_ordinal()
        print_fn(f"TPU/XLA detected ({world_size} cores)")
    else:
        device = default_device
        print_fn = print
        world_size = 1
        rank = 0
        print_fn(f"Using device: {device}")
        if device.type == 'cuda':
            print_fn(f"GPU: {torch.cuda.get_device_name(0)}")
            print_fn(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB total")

    torch.manual_seed(seed + rank)
    print_fn(f"Vocab size: {_vocab_size}")
    print_fn(f"Effective batch size: {batch_size * world_size}")

    # Move data to device
    train_data = _train_data.to(device)
    val_data = _val_data.to(device)

    model = build_model(_vocab_size).to(device)
    
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
    )

    checkpoint_path = 'checkpoint.pt'
    start_iter = 0
    if os.path.exists(checkpoint_path):
        print_fn(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        try:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            _move_optimizer_state(optimizer, device)
            start_iter = checkpoint['iter'] + 1
            print_fn(f"Resumed from iteration {checkpoint['iter']} (train_loss: {checkpoint['train_loss']:.4f}, val_loss: {checkpoint['val_loss']:.4f})")
        except RuntimeError as exc:
            print_fn(f"Checkpoint incompatible with current model, starting fresh. ({exc})")
    else:
        print_fn("Starting fresh training...")

    for iter in range(start_iter, max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, device, train_data, val_data)
            print_fn(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            if _is_master():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'lr': lr,
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                }
                _save_checkpoint(checkpoint, checkpoint_path)
                print_fn(f"Saved checkpoint at step {iter}")

        xb, yb = get_batch('train', device, train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if use_xla:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

    print_fn("\n" + "="*80)
    print_fn("Training complete!")
    print_fn(f"Final checkpoint saved to: {checkpoint_path}")
    print_fn("="*80)
    print_fn("\nTo generate text, run: python generate.py")
    print_fn("For more options: python generate.py --help")

def main():
    global _train_data, _val_data, _vocab_size
    
    # Load data in main process only
    _log_initial_device()
    _vocab_size = _load_data()
    print(f"Vocab size: {_vocab_size}")
    
    if use_xla and xmp is not None:
        # For TPU: use spawn to create worker processes
        xmp.spawn(_train_worker, args=())
    else:
        _train_worker(0)

if __name__ == "__main__":
    main()
