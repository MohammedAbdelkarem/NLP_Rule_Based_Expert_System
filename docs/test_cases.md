# Manual Test Cases

## Case 1 — Input Encoding only

Expected: valid.

```text
Input Encoding Layer Group
```

## Case 2 — Input Encoding → LayerNorm

Expected: valid.

```text
Input Encoding → LayerNorm
```

## Case 3 — Input Encoding → Multi-Head Attention

Expected: valid if `model_dim % num_heads == 0`.

```text
Input Encoding → Multi-Head Attention
```

## Case 4 — Multi-Head Attention first with raw token IDs

Expected: rollback.

## Case 5 — Invalid number of attention heads

Example:

```text
model_dim = 250
num_heads = 8
```

Expected: ask again because `model_dim` is not divisible by `num_heads`.

## Case 6 — Residual Connection too early

Expected: rollback.

```text
Input Encoding → Residual Connection
```

## Case 7 — Valid Residual Connection

Expected: valid if source shape equals previous output shape.

```text
Input Encoding → Multi-Head Attention → Residual Connection
```

## Case 8 — Feed-Forward after hidden tensor

Expected: valid.

```text
Input Encoding → Feed-Forward Network
```

The system should ask for:

```text
d_ff
activation: ReLU or GELU
```

## Case 9 — Feed-Forward first with hidden tensor

Expected: valid.

The system asks for:

```text
input kind
batch_size
sequence_length
model_dim
d_ff
activation
```

## Case 10 — Feed-Forward first with raw token IDs

Expected: rollback.

## Case 11 — ReLU standalone layer

Expected: valid after any float hidden tensor.

```text
Input Encoding → ReLU
```

## Case 12 — GELU standalone layer

Expected: valid after any float hidden tensor.

```text
Input Encoding → GELU
```

## Case 13 — Activation after activation

Expected: rollback.

```text
ReLU → GELU
```

or

```text
GELU → ReLU
```

## Case 14 — Full Transformer-style partial block

Expected: valid.

```text
Input Encoding
→ LayerNorm
→ Multi-Head Attention
→ Residual Connection
→ LayerNorm
→ Feed-Forward Network
→ Residual Connection
```


## Test Case: Position-wise Linear after Input Encoding

```text
1. Input Encoding Layer Group
2. Position-wise Linear Layer

Input Encoding values:
batch_size = 2
sequence_length = 5
vocab_size = 1000
model_dim = 64

Linear values:
output_dim = 128
purpose = Hidden projection

Expected output shape:
(2, 5, 128)

Expected data_type:
float_hidden_tensor
```

## Test Case: Position-wise Linear as logits projection

```text
Previous output shape:
(2, 5, 64)

Linear values:
output_dim = 1000
purpose = Output logits projection

Expected output shape:
(2, 5, 1000)

Expected data_type:
logits
```


## Test Case: Position-wise Linear logits projection → Softmax

Expected: valid.

```text
Input Encoding
→ Position-wise Linear Layer
→ Softmax Layer
```

Linear purpose:

```text
Output score projection before Softmax
```

Expected Softmax behavior:

```text
input data_type = logits
output data_type = probabilities
shape unchanged
```

## Test Case: Softmax after hidden tensor

Expected: rollback.

```text
Input Encoding → Softmax
```

Reason:

```text
Softmax requires logits, not ordinary float_hidden_tensor.
```

## Test Case: Softmax first with logits tensor

Expected: valid.

The system asks for:

```text
input kind = Logits tensor
batch_size
sequence_length
logits vector dimension
```

## Test Case: Softmax first with raw token IDs

Expected: rollback.

## Test Case: Refactored activation structure

Expected: activation rules remain logically identical after refactor.

```text
ReLU, GELU, and Softmax are separate activation folders.
ReLU/GELU shared declaration logic is kept in activations/common.
Softmax stays an activation layer but requires logits and ends the session.
```

## Test Case: Transformer chapter coverage

Expected: core Transformer components are represented.

```text
Input Encoding = Token Embedding + Positional Embedding + Summation
Transformer Block = LayerNorm + Multi-Head Attention + Residual + Feed-Forward
Output path = Position-wise Linear logits projection + Softmax
```

## Test Case: Refactored rule folders

Expected: imports and engine construction remain valid after folder cleanup.

```text
src/rules/transformers/
src/rules/activations/
src/rules/projections/
```

There should be no remaining `source_transformers` or `source_nlp` paths.

## Test Case: Refactored Linear structure

Expected: Position-wise Linear remains logically identical after refactor.

```text
src/rules/projections/linear/parameter_rules.py
src/rules/projections/linear/calculation_rules.py
src/rules/projections/linear/output_rules.py
src/rules/projections/compatibility/linear.py
```

The layer should still support:

```text
hidden projection → float_hidden_tensor
output score projection before Softmax → logits
```
