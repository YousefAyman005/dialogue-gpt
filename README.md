# dialogue-gpt 🎬💬

A GPT-style transformer model trained on movie and TV show dialogue from the Cornell Movie Dialogs Corpus. This model learns to generate realistic conversations between characters, understanding genre context and character dynamics.

## ✨ Features

- **Character-aware dialogue generation**: Generates conversations between specific characters
- **Genre conditioning**: Supports different movie/TV genres (drama, comedy, romance, etc.)
- **Multi-GPU training**: Distributed Data Parallel (DDP) support for efficient training on multiple GPUs
- **TPU support**: Compatible with Google Colab and Kaggle TPU instances
- **Efficient tokenization**: Uses tiktoken (GPT-2 tokenizer) for robust text encoding
- **Early stopping & checkpointing**: Automatically saves best models and prevents overfitting

## 🏗️ Architecture

- **Model**: Transformer decoder (GPT-style)
- **Tokenizer**: tiktoken (GPT-2 encoding, vocab_size=50257)
- **Default hyperparameters**:
  - Embedding dimension: 384
  - Attention heads: 6
  - Layers: 6
  - Block size: 256
  - Dropout: 0.2

## 🚀 Quick Start

### Installation

```bash
pip install torch tiktoken kagglehub
```

### Prepare Data

```bash
python scripts/make_cornell_input.py
```

This downloads the Cornell Movie Dialogs Corpus and formats it for training.

### Training

**Single GPU/CPU:**
```bash
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --standalone --nproc_per_node=2 train.py
```

**Custom hyperparameters:**
```bash
export BATCH_SIZE=32
export BLOCK_SIZE=256
export N_EMBD=384
export N_HEAD=6
export N_LAYERS=6
export LEARNING_RATE=3e-4
python train.py
```

### Generating Dialogue

**Basic generation:**
```bash
python generate.py
```

**With custom prompt:**
```bash
python generate.py --prompt "GENRES=drama
CHARACTERS=ALICE|BOB
ALICE: Hello!" --tokens 300
```

**Adjust creativity:**
```bash
python generate.py --temperature 0.8  # More focused (default: 1.0)
python generate.py --temperature 1.2  # More creative
```

## 📊 Training Results

The model was trained on 2x Tesla T4 GPUs (Kaggle) for ~5000 steps:

- **Best validation loss**: 2.96 (step 3000-5000)
- **Training time**: ~1.5 hours
- **Final train loss**: 2.46 
- **Dataset**: Cornell Movie Dialogs Corpus (~220k conversational exchanges)

## 🎭 Sample Output

GENRES=drama|mystery|romance|sci-fi
CHARACTERS=SID|ZACK
SID: Any ideas Paul get out of here.
ZACK: Don't worry. You're non-with it.
<|endoftext|>
GENRES=drama|mystery|romance|sci-fi
CHARACTERS=KELVIN|SID
KELVIN: I'm not sure I'll help you get to sleep.
SID: I understand perfectly. I'll be here.
<|endoftext|>
GENRES=drama|mystery|romance|sci-fi
CHARACTERS=SID|KELVIN
SID: Hello.
KELVIN: Yes? I'm here. Why did you do it?
SID: Because I have a whole good time crying gag. I turned down the floor with the windows crawling down there. You know plenty of Hollis got to calm down. And now your father makes me look like we have a wit and an coop on me. It's like his own thing. You you can afford thinking about how many times I've seen before. If you let those bastards slapped the tips.
<|endoftext|>

## 🛠️ Project Structure

```
dialogue-gpt/
├── train.py              # Training script with DDP support
├── generate.py           # Inference script
├── data/
│   └── movies/
│       └── input.txt     # Formatted training data
├── scripts/
│   └── make_cornell_input.py  # Data preparation script
└── checkpoint.pt         # Saved model checkpoint
```

## 📝 Data Format

The model expects input in this format:

```
GENRES=drama
CHARACTERS=CHARACTER1|CHARACTER2
CHARACTER1: dialogue line
CHARACTER2: dialogue line
<|endoftext|>
GENRES=comedy
CHARACTERS=CHARACTER3|CHARACTER4
...
```

## 🎯 Configuration

All hyperparameters can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | 64 | Training batch size |
| `BLOCK_SIZE` | 256 | Context window size |
| `N_EMBD` | 384 | Embedding dimension |
| `N_HEAD` | 6 | Number of attention heads |
| `N_LAYERS` | 6 | Number of transformer layers |
| `FFN_MULT` | 4 | FFN hidden layer multiplier |
| `DROPOUT` | 0.2 | Dropout rate |
| `LEARNING_RATE` | 3e-4 | Initial learning rate |
| `MAX_ITERS` | 20000 | Maximum training iterations |
| `EVAL_INTERVAL` | 1000 | Steps between evaluations |

## 🔧 Advanced Usage

### Training on Kaggle with Multiple GPUs

```python
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["BATCH_SIZE"] = "16"
os.environ["BLOCK_SIZE"] = "128"

!torchrun --standalone --nproc_per_node=2 train.py
```

### Loading and Using a Checkpoint

```python
import torch
import tiktoken
from train import BigramLanguageModel

# Load checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('checkpoint.pt', map_location=device)

# Initialize model
model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate
enc = tiktoken.get_encoding("gpt2")
prompt = "GENRES=drama\nCHARACTERS=ALICE|BOB\nALICE: Hello!\nBOB: "
context = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=200)
    print(enc.decode(output[0].tolist()))
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Share interesting generated dialogues

## 🙏 Acknowledgments

- Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- Training data from [Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- Inspired by the transformer architecture from "Attention Is All You Need"

## 📧 Contact

For questions or collaborations, feel free to open an issue!

what do u think of this readme if u coulf provide a better that would be better