# Neural Network Expert System — NLP Layers

This project is a console-based Knowledge-Based System built with Experta. It helps the user build NLP neural-network layers step by step while checking tensor shape, tensor data type, layer order, and layer-specific restrictions.

The system is not an automatic architecture generator. The user chooses the layer order. The expert system validates each choice, calculates the output shape, stores the layer facts, and rolls back invalid choices with a correction suggestion when needed.

## Available layers

1. Input Encoding Layer Group
2. LayerNorm
3. Multi-Head Attention Layer Group
4. Residual Connection Layer
5. Feed-Forward Network Layer Group
6. ReLU Activation Layer
7. GELU Activation Layer
8. Position-wise Linear Layer
9. Softmax Layer
10. Audio Input Layer (Speech Module)
11. Audio Framing + Windowing Layer (Speech Module)

## Current tensor notation

The project uses practical batch notation:

```text
(batch_size, sequence_length, model_dim)
```

The chapter notation `[N × d]` is the theoretical single-sequence version. This project uses `(B, N, d)` for implementation-oriented shape reasoning.

## Transformer chapter coverage

| Transformer component | System implementation |
|---|---|
| Token Embedding | inside Input Encoding Layer Group |
| Positional Embedding | inside Input Encoding Layer Group |
| Element-wise Embedding Sum | inside Input Encoding Layer Group |
| LayerNorm | standalone Transformer layer |
| Multi-Head Attention | grouped Transformer layer |
| Q/K/V projections | internal tensors inside Multi-Head Attention |
| Attention Softmax | internal tensor step inside Multi-Head Attention |
| Attention Output Projection | internal tensor step inside Multi-Head Attention |
| Residual Connection | standalone Transformer layer |
| Feed-Forward Network | grouped Transformer layer |
| FFN ReLU/GELU choice | internal activation choice inside Feed-Forward Network |
| Linear / Dense Projection | Position-wise Linear Layer |
| Output Scores / Logits | Position-wise Linear purpose option |
| Softmax | standalone activation layer that converts logits to probabilities |

A separate Language Modeling Head layer is not implemented as its own menu item yet. Its basic flow is currently represented by:

```text
Position-wise Linear Layer with output score projection
→ Softmax Layer
```

## Main design decisions

- `LayerParameter` stores user-entered numeric values.
- `TransformationSuggestion` is kept for adapter or correction suggestions.
- `OutputStatus` prevents duplicate output actions.
- `TensorShape` and `TensorDataType` were merged into `TensorInfo`.
- `IntermediateTensor` is used only for grouped or complex layers.
- Transformer-specific rules are under `src/rules/transformers/`.
- Speech-specific rules are under `src/rules/speech/`.
- Printing helpers were split by module under `src/utils/printing/`; `src/utils/print_helpers.py` remains as a compatibility facade.
- ReLU, GELU, and Softmax are normal selectable activation layers under `src/rules/activations/`.
- Feed-Forward Network is a grouped Transformer layer and asks the user whether to use ReLU or GELU internally.
- Position-wise Linear is a standalone projection layer under `src/rules/projections/`.
- Position-wise Linear can output either hidden tensors or logits depending on its purpose.
- Softmax is stored as an activation layer and remains terminal because it outputs probabilities.

## Structure

```text
src/
├── facts/
├── rules/
│   ├── common/
│   ├── transformers/
│   │   ├── compatibility/
│   │   ├── input_encoding/
│   │   ├── layer_norm/
│   │   ├── multi_head_attention/
│   │   ├── residual_connection/
│   │   └── feed_forward/
│   ├── activations/
│   │   ├── common/
│   │   │   └── activation_declaration_helpers.py
│   │   ├── compatibility/
│   │   │   ├── relu_gelu.py
│   │   │   └── softmax.py
│   │   ├── relu/
│   │   │   ├── parameter_rules.py
│   │   │   ├── calculation_rules.py
│   │   │   └── output_rules.py
│   │   ├── gelu/
│   │   │   ├── parameter_rules.py
│   │   │   ├── calculation_rules.py
│   │   │   └── output_rules.py
│   │   └── softmax/
│   │       ├── parameter_rules.py
│   │       ├── calculation_rules.py
│   │       └── output_rules.py
│   ├── projections/
│   │   ├── compatibility/
│   │   │   └── linear.py
│   │   └── linear/
│   │       ├── parameter_rules.py
│   │       ├── calculation_rules.py
│   │       └── output_rules.py
│   └── speech/
│       ├── compatibility/
│       │   ├── audio_input.py
│       │   └── framing_windowing.py
│       ├── audio_input/
│       │   ├── parameter_rules.py
│       │   ├── calculation_rules.py
│       │   └── output_rules.py
│       └── framing_windowing/
│           ├── parameter_rules.py
│           ├── calculation_rules.py
│           └── output_rules.py
├── utils/
│   ├── print_helpers.py
│   └── printing/
│       ├── common.py
│       ├── transformers.py
│       ├── activations.py
│       ├── projections.py
│       └── speech.py
├── engine.py
└── prompts.py
```

## Feed-Forward Network

Feed-Forward Network is a grouped Transformer layer:

```text
Input:  (B, N, d)
First projection: (B, N, d) → (B, N, d_ff)
Activation: ReLU or GELU, shape unchanged
Second projection: (B, N, d_ff) → (B, N, d)
Output: (B, N, d)
```

Trainable parameters:

```text
W1: (d, d_ff)
b1: (d_ff,)
W2: (d_ff, d)
b2: (d,)
```

## Activation layers

ReLU and GELU are standalone simple layers:

```text
input data_type = float_hidden_tensor
output data_type = float_hidden_tensor
input shape = output shape
no trainable parameters
layer_role = activation
terminal_status = non_terminal
```

Softmax is also stored under activations:

```text
input data_type = logits
output data_type = probabilities
input shape = output shape
no trainable parameters
layer_role = activation
terminal_status = terminal
```

The system blocks direct activation after activation, such as:

```text
ReLU → GELU
GELU → ReLU
ReLU → ReLU
GELU → GELU
Softmax → any next layer because Softmax is terminal
```

## Position-wise Linear Layer

The Position-wise Linear Layer is a general NLP projection layer. It changes the last tensor dimension:

```text
(B, N, input_dim) → (B, N, output_dim)
```

It asks for the output dimension and layer purpose:

```text
1. Hidden projection → output data_type = float_hidden_tensor
2. Output score projection before Softmax → output data_type = logits
```

Stored parameter shapes:

```text
W: (input_dim, output_dim)
b: (output_dim,)
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written to:

```text
outputs/calculation_trace.txt
outputs/session_summary.txt
```


## Speech Module: Audio Input Layer

The first speech layer represents raw audio/WAV information as a numeric waveform tensor while preserving the channel dimension:

```text
(batch_size, samples_per_channel, channels)
```

The system asks for:

```text
batch_size
duration_seconds
sample_rate
channels
bit_depth
```

Shape calculation:

```text
samples_per_channel = duration_seconds × sample_rate
output_shape = (batch_size, samples_per_channel, channels)
```

This means mono, stereo, and multi-channel audio are all accepted without forcing the user to downmix to mono.

## Speech Module: Audio Framing + Windowing Layer

The second speech layer splits the raw waveform into short overlapping frames and applies a window function:

```text
Input:  (batch_size, samples_per_channel, channels)
Output: (batch_size, num_frames, frame_length_samples, channels)
```

The system asks for:

```text
frame_length_ms
hop_length_ms
window_function = Hann / Hamming / Rectangular
```

Shape calculation:

```text
frame_length_samples = sample_rate × frame_length_ms / 1000
hop_length_samples = sample_rate × hop_length_ms / 1000
num_frames = 1 + floor((samples_per_channel - frame_length_samples) / hop_length_samples)
output_shape = (batch_size, num_frames, frame_length_samples, channels)
```

Example:

```text
Audio Input output = (1, 32000, 2)
sample_rate = 16000
frame_length_ms = 25
hop_length_ms = 10

frame_length_samples = 400
hop_length_samples = 160
num_frames = 198
Framing + Windowing output = (1, 198, 400, 2)
```

The channel dimension is preserved, so the layer supports mono, stereo, and multi-channel audio.
