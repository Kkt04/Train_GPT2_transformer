GPT Decoder-Only Transformer
A PyTorch implementation of a GPT-style decoder-only transformer for character-level language modeling.

Model Architecture
This implementation follows the GPT-2 architecture with the following components:

Decoder-only Transformer: 4 layers of transformer blocks

Multi-head Self-attention: 4 attention heads per layer

Embedding Dimension: 64

Context Length: 32 tokens

Vocabulary: Character-level based on the Shakespeare dataset

Key Components:
Token Embeddings: Learnable embeddings for each character

Positional Embeddings: Learnable positional encodings

Transformer Blocks (4 layers):

Multi-head self-attention with causal masking

Feed-forward network with ReLU activation

Layer normalization and residual connections

Final Layer Norm: Applied before the output layer

Language Model Head: Linear projection to vocabulary size

Training Details
Batch Size: 16

Learning Rate: 1e-3

Training Steps: 5000

Evaluation: Every 100 steps

Optimizer: AdamW

Results
The model achieves:

Training loss: ~1.66

Validation loss: ~1.82

After 5000 training steps, the model generates coherent Shakespeare-like text with proper character names, dialogue structure, and poetic language patterns.

Usage
python
# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)
Files
gpt_decoder.py: Main implementation file

input.txt: Shakespeare training dataset

Model parameters: ~209,729 parameters (0.21M)

Requirements
PyTorch

CUDA (optional, for GPU acceleration)

This implementation demonstrates the core principles of modern decoder-only transformer architectures used in models like GPT-2, GPT-3, and beyond.
