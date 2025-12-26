import torch
from train import model, decode, encode, device

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()
print(f"Loaded model from iteration {checkpoint['iter']}")

# Generate text
prompt = ""  # Add your prompt here, e.g., "JOHN: Hello!"
max_tokens = 500

if prompt:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=max_tokens)
    print(decode(output[0].tolist()))
