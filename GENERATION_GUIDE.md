# Generate Text with dialogue-gpt

After training your model, use `generate.py` to create text samples.

## Basic Usage

### Simple Generation
```bash
python generate.py
```
This generates 500 tokens unconditionally (no prompt).

### With a Prompt
```bash
python generate.py --prompt "JOHN: Hello there!"
```

### Control Length
```bash
python generate.py --tokens 1000
```

### Adjust Temperature
Temperature controls randomness (0.1 = conservative, 2.0 = creative):
```bash
python generate.py --temperature 0.8  # More focused
python generate.py --temperature 1.5  # More creative
```

### Top-K Sampling
Only sample from the top K most likely tokens:
```bash
python generate.py --top_k 40
```

### Multiple Samples
Generate multiple samples at once:
```bash
python generate.py --num_samples 3
```

## Full Example

```bash
python generate.py \
  --prompt "ALICE: How are you?" \
  --tokens 300 \
  --temperature 0.9 \
  --top_k 50 \
  --num_samples 2
```

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | `""` | Starting text (empty for unconditional) |
| `--tokens` | `500` | Number of tokens to generate |
| `--temperature` | `0.8` | Sampling temperature (0.1-2.0) |
| `--top_k` | `None` | Top-k sampling (try 40-50) |
| `--num_samples` | `1` | Number of samples to generate |

## Tips

### Best Settings for Dialogue
```bash
python generate.py \
  --prompt "CHARACTER_NAME: " \
  --temperature 0.9 \
  --top_k 50 \
  --tokens 300
```

### More Coherent Output
- Lower temperature (0.7-0.9)
- Enable top_k (40-60)

### More Creative Output
- Higher temperature (1.2-1.5)
- Disable top_k or use higher value (100+)

### Exploring the Model
- No prompt + high temperature = see what the model learned
- Specific character names in prompt = character-specific dialogue

## Programmatic Usage

You can also import and use the generation function in your own Python scripts:

```python
from generate import generate_text

# Generate text
output = generate_text(
    prompt="SARAH: Good morning!",
    max_tokens=200,
    temperature=0.8,
    top_k=50
)

print(output)
```

## Troubleshooting

### "Checkpoint file not found"
Train the model first:
```bash
python train.py
```

### Out of Memory
Reduce `--tokens`:
```bash
python generate.py --tokens 200
```

### Poor Quality Output
- Model needs more training
- Try different temperature values
- Use top_k sampling for better quality
