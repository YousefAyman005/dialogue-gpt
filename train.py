import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    _xla_available = True
except ImportError:
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
batch_size = _env_int("BATCH_SIZE", 32) # per-core batch size
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
n_layers = _env_int("N_LAYERS", 8)
ffn_mult = _env_int("FFN_MULT", 6)
dropout = _env_float("DROPOUT", 0.2)
tpu_cores = _env_int("TPU_NUM_CORES", 8)
seed = _env_int("SEED", 1337)
# ------------

if n_embd % n_head != 0:
    raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

def _should_use_xla():
    if not _xla_available:
        return False
    if os.environ.get("USE_TPU") == "1":
        return True
    return any(os.environ.get(name) for name in ("COLAB_TPU_ADDR", "TPU_NAME", "XRT_TPU_CONFIG"))

use_xla = _should_use_xla()
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = default_device

with open('data/movies/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize tiktoken encoder (using GPT-2 encoding)
enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
decode = lambda l: enc.decode(l)
vocab_size = enc.n_vocab
print(f"Vocab size: {vocab_size}")

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split, device)
            logits, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    # Linear warmup then cosine decay.
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
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (q.size(-1) ** -0.5) # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)

        out = wei @ v # (B,T,head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd, ffn_mult)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x    
# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[blocks(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def build_model():
    return BigramLanguageModel()

if __name__ != "__main__":
    model = build_model().to(device)

def _is_master():
    return (not use_xla) or xm.is_master_ordinal()

def _save_checkpoint(checkpoint, path):
    if use_xla:
        xm.save(checkpoint, path)
    else:
        torch.save(checkpoint, path)

def _train_worker(index, num_cores):
    if use_xla:
        device = xm.xla_device()
        print_fn = xm.master_print
        world_size = num_cores
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
    print_fn(f"Effective batch size: {batch_size * world_size}")

    model = build_model().to(device)
    # create a PyTorch optimizer
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
    )

    # Load checkpoint if exists
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

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, device)
            print_fn(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            # Save checkpoint
            if _is_master():
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'lr': lr,
                    'train_loss': losses['train'].item(),
                    'val_loss': losses['val'].item(),
                }
                _save_checkpoint(checkpoint, checkpoint_path)
                print_fn(f"Saved checkpoint at step {iter}")

        # sample a batch of data
        xb, yb = get_batch('train', device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if use_xla:
            xm.optimizer_step(optimizer, barrier=True)
            xm.mark_step()
        else:
            optimizer.step()

    print_fn("\n" + "="*80)
    print_fn("Training complete!")
    print_fn(f"Final checkpoint saved to: {checkpoint_path}")
    print_fn("="*80)
    print_fn("\nTo generate text, run: python generate.py")
    print_fn("For more options: python generate.py --help")

if __name__ == "__main__":
    if use_xla and xmp is not None:
        xmp.spawn(_train_worker, args=(tpu_cores,), nprocs=tpu_cores, start_method='fork')
    else:
        _train_worker(0, 1)
