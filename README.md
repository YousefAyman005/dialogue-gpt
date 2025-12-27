# dialogue-gpt
A character-conditioned GPT-style Transformer trained on movie and TV show dialogue, based on the nanoGPT architecture.

## Quick Start

### Training
```bash
pip install torch tiktoken kagglehub
python scripts/make_cornell_input.py
python train.py
```

### Generating Text
```bash
python generate.py --prompt "ALICE: Hello!" --tokens 300 --temperature 0.9
```

See [GENERATION_GUIDE.md](GENERATION_GUIDE.md) for detailed generation options.
