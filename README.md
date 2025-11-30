# GPT Decoder-Only Transformer

A PyTorch implementation of a GPT-style **decoder-only transformer** for character-level language modeling.

---

## 🔧 Model Architecture

This implementation follows the GPT-2 style architecture with the following components:

### **Key Specs**
- **Decoder-only Transformer:** 4 layers  
- **Multi-head Self-attention:** 4 heads per layer  
- **Embedding Dimension:** 64  
- **Context Length:** 32 tokens  
- **Vocabulary:** Character-level (Shakespeare dataset)

### **Components**
#### ⭐ Token Embeddings  
Learnable embeddings for each character.

#### ⭐ Positional Embeddings  
Learnable positional encodings added to token embeddings.

#### ⭐ Transformer Blocks (4 Layers)
Each block contains:
- Multi-head self-attention with **causal masking**
- Feed-forward network with **ReLU** activation
- Residual connections
- Layer normalization

#### ⭐ Final Layer Norm  
Applied before the output projection.

#### ⭐ Language Model Head  
A linear layer projecting hidden states → vocabulary logits.

---

## 🚀 Training Details

- **Batch Size:** 16  
- **Learning Rate:** 1e-3  
- **Training Steps:** 5000  
- **Evaluation:** Every 100 steps  
- **Optimizer:** AdamW  

### 📉 Results
- **Training Loss:** ~1.66  
- **Validation Loss:** ~1.82  
- Generates coherent Shakespeare-like text after training, including:
  - Character names  
  - Dialogue structures  
  - Poetic rhythm and patterns  

---

## 🧪 Usage

```python
# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)

📁 Files

gpt_decoder.py — Main model implementation

input.txt — Shakespeare training dataset

Model Parameters: ~209,729 (~0.21M)

📦 Requirements

PyTorch

CUDA (optional, for GPU acceleration)
