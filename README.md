# Transformer Architecture: "Attention is All You Need"

## Project Overview
This project implements the complete Transformer neural network architecture as described in the seminal research paper **"Attention Is All You Need"** by Vaswani et al. (2017). The implementation provides a modular, thoroughly tested foundation for sequence-to-sequence learning tasks, with particular focus on neural machine translation applications. All components have been built from scratch following software engineering best practices and include comprehensive documentation and testing.

**Project Date:** 11/11/2025  
**Developers:** 
- Muhammad Usman Akmal (IKHJLS)
- Ekka Arpan (Y6LC1D)

## Architecture Overview

The Transformer represents a paradigm shift in sequence modeling by relying entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers. This architecture enables parallel processing of entire sequences while effectively capturing long-range dependencies through self-attention mechanisms.

### Core Innovation Points
- **Self-Attention Mechanisms**: Replace recurrence with scaled dot-product attention
- **Parallel Processing**: Entire sequences processed simultaneously
- **Positional Encoding**: Inject sequence order information without recurrence
- **Multi-Head Attention**: Capture different representation subspaces in parallel

## Component Specifications

### 1. Scaled Dot-Product Attention
The core attention mechanism computes compatibility scores between query and key vectors, then uses these scores to create weighted combinations of value vectors.

**Key Characteristics:**
- Implements the standard attention formulation with proper scaling
- Includes optional masking capabilities for sequence processing
- Maintains numerical stability through appropriate scaling factors
- Returns both context vectors and attention weights for interpretability

### 2. Multi-Head Attention
**Architectural Design:**
This component projects input sequences into multiple representation subspaces, allowing the model to jointly attend to information from different perspectives.

**Implementation Features:**
- Parallel computation across multiple attention heads
- Linear transformations for query, key, and value projections
- Concatenation and final linear projection of head outputs
- Dimension validation to ensure divisibility constraints

### 3. Positional Encoding
Since the Transformer contains no recurrence or convolution, positional encodings inject information about the relative or absolute position of tokens in the sequence.

**Encoding Strategy:**
- Uses sinusoidal functions of different frequencies
- Alternating sine and cosine functions for even and odd positions
- Precomputed encoding matrix for efficiency
- Dropout application for regularization

### 4. Position-wise Feed-Forward Networks
A fully connected feed-forward network applied independently and identically to each position.

**Structural Details:**
- Two linear transformations with ReLU activation
- Inner layer expansion factor of 4 (as per original paper)
- Dropout for regularization between layers
- Position-independent application

### 5. Encoder Components
**Encoder Layer Composition:**
Each encoder layer contains two main sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise feed-forward network

**Normalization and Connections:**
- Residual connections around each sub-layer
- Layer normalization applied after residual connections
- Dropout for regularization during training

**Encoder Stack:**
Multiple identical encoder layers stacked to form the complete encoder, processing source sequences in parallel.

### 6. Decoder Components
**Decoder Layer Composition:**
Each decoder layer contains three main sub-layers:
1. Masked multi-head self-attention (target sequence)
2. Multi-head encoder-decoder attention (cross-attention)
3. Position-wise feed-forward network

**Masking Strategy:**
- Look-ahead mask prevents attending to future tokens
- Padding mask handles variable-length sequences
- Combined masking for training efficiency

### 7. Complete Transformer Model
**System Integration:**
The full model integrates all components into a cohesive architecture:
- Source and target embedding layers
- Positional encoding injections
- Encoder and decoder stacks
- Final linear projection layer

**Advanced Features:**
- Weight sharing between embedding and output layers
- Proper parameter initialization (Xavier uniform)
- Comprehensive mask generation for sequences
- Configurable hyperparameters match original specifications

## Implementation Methodology

### Software Engineering Practices

**Modular Design Approach:**
Each architectural component is implemented as an independent, reusable module with clear interfaces and well-defined responsibilities.

**Code Quality Measures:**
- Comprehensive documentation strings for all functions and classes
- Type hints throughout for enhanced code clarity and IDE support
- Detailed inline comments explaining complex operations and mathematical formulations
- Consistent coding style and naming conventions

**Error Handling and Validation:**
- Assertion checks for parameter validation
- Dimension verification throughout forward passes
- Graceful handling of edge cases
- Informative error messages for debugging

### Testing Strategy

**Unit Testing Coverage:**
Individual component testing includes:
- Shape consistency verification across operations
- Mask functionality validation
- Attention weight distribution checks
- Numerical value verification for deterministic components

**Integration Testing:**
End-to-end testing validates:
- Complete forward pass functionality
- Component interoperability
- Shape preservation through entire network
- Multi-layer stack propagation

**Test-Driven Development:**
Standalone test blocks within each module allow for immediate verification during development and maintenance.

## Model Specifications

The implementation adheres to the original paper's specifications:

| Component | Parameter | Value | Significance |
|-----------|-----------|-------|--------------|
| Model Dimensions | d_model | 512 | Base embedding dimension |
| Attention | Heads (h) | 8 | Parallel attention mechanisms |
| Feed-Forward | d_ff | 2048 | Inner layer expansion |
| Architecture | Layers | 6 | Encoder/decoder depth |
| Regularization | Dropout | 0.1 | Prevent overfitting |
| Sequences | max_len | 5000 | Maximum sequence length |

## Usage

```python
# Example usage
from transformer import Transformer

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=5000,
    dropout=0.1
)

# Forward pass
output = model(source_sequence, target_sequence)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- (Additional dependencies as needed for your implementation)

## Project Structure

```
transformer-project/
├── src/
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   └── transformer.py
├── tests/
│   ├── test_attention.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_transformer.py
├── examples/
│   └── translation_example.py
└── README.md
```
```markdown


To run the unit tests for this project, use the following command:

```bash
pytest -v
```

This will run all test cases and display verbose output showing which tests passed/failed.

## Test Structure
- Tests are located in the `tests/` directory
- Layer tests: `test_layers.py`
- Model tests: `test_model.py`

## Expected Output
When tests pass successfully, you should see:
```
10 passed in X.XXs
```

All 10 test cases should pass, covering:
- Scaled Dot Product Attention (with and without mask)
- Positional Encoding
- Multi-Head Attention
- Position-wise Feed Forward
- Encoder Layer
- Encoder
- Transformer Forward Pass
```

## References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

