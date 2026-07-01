MAIN_MENU = """
Available layers:
1. Input Encoding Layer Group: Token Embedding + Positional Embedding + Element-wise Summation
2. LayerNorm
3. Multi-Head Attention Layer Group
4. Residual Connection Layer
5. Feed-Forward Network Layer Group
6. ReLU Activation Layer
7. GELU Activation Layer
8. Position-wise Linear Layer
9. Softmax Layer
"""

CHOOSE_FIRST_LAYER = "\nChoose the first layer number: "
CHOOSE_NEXT_LAYER = "\nChoose the next layer number: "
ASK_ROLLBACK_LAYER = "\nChoose another layer number for the same layer position, or 0 to end the session: "

INPUT_ENCODING_SELECTED = """
You selected: Input Encoding Layer Group

This is not one simple layer.
It is a grouped Transformer input layer made of two internal layers working together:
1. Token Embedding layer
2. Positional Embedding layer

How they work:
- Token Embedding uses the raw token IDs.
- Positional Embedding uses the position indices of those tokens.
- Both internal layers produce float tensors with the same shape.
- Their outputs are combined using element-wise summation to create Transformer input X.

Enter the raw token input information:
"""

ASK_BATCH_SIZE = "Enter batch size: "
ASK_SEQUENCE_LENGTH = "Enter sequence length / context length: "
ASK_VOCAB_SIZE = "Enter vocabulary size / number of tokens: "
ASK_MODEL_DIM = "Enter model dimension d_model / embedding vector size: "
ASK_NUM_HEADS = "Enter number of attention heads: "
ASK_RESIDUAL_SOURCE_LAYER_INDEX = "Enter residual source layer index to add with the previous layer output: "
ASK_FEED_FORWARD_DIM = "Enter feed-forward hidden dimension d_ff: "
ASK_INPUT_DIM = "Enter input feature dimension: "
ASK_POSITION_WISE_LINEAR_OUTPUT_DIM = "Enter output feature dimension: "
ASK_SOFTMAX_VECTOR_DIM = "Enter logits vector dimension / number of scores: "

LAYER_NORM_SELECTED = """
You selected: LayerNorm Layer

Because LayerNorm is selected as the first layer, your input must already be an encoded float tensor.
LayerNorm cannot be applied directly to raw integer token IDs.
"""

ASK_LAYER_NORM_INPUT_KIND = """
Choose your input data kind:
1. Encoded / hidden float tensor
2. Raw token IDs

Your choice: """

MULTI_HEAD_ATTENTION_SELECTED = """
You selected: Multi-Head Attention Layer Group

Because Multi-Head Attention is selected as the first layer, your input must already be an encoded float tensor.
Multi-Head Attention cannot be applied directly to raw integer token IDs.
"""

ASK_MULTI_HEAD_ATTENTION_INPUT_KIND = """
Choose your input data kind:
1. Encoded / hidden float tensor
2. Raw token IDs

Your choice: """

RESIDUAL_CONNECTION_SELECTED = """
You selected: Residual Connection Layer

Residual Connection adds the previous layer output to an earlier compatible tensor.
The two tensors must have the same shape and compatible data type.
"""

FEED_FORWARD_SELECTED = """
You selected: Feed-Forward Network Layer Group

Because Feed-Forward Network is selected as the first layer, your input must already be an encoded float tensor.
Feed-Forward Network cannot be applied directly to raw integer token IDs.
"""

ASK_FEED_FORWARD_INPUT_KIND = """
Choose your input data kind:
1. Encoded / hidden float tensor
2. Raw token IDs

Your choice: """

ASK_FEED_FORWARD_ACTIVATION = """
Choose the activation function inside Feed-Forward Network:
1. ReLU
2. GELU

Your choice: """

ACTIVATION_SELECTED = """
You selected an Activation Layer

Because an activation layer is selected as the first layer, your input must already be an encoded float tensor.
Activation layers cannot be applied directly to raw integer token IDs.
"""

POSITION_WISE_LINEAR_SELECTED = """
You selected: Position-wise Linear Layer

Because Position-wise Linear is selected as the first layer, your input must already be a float tensor.
Position-wise Linear cannot be applied directly to raw integer token IDs.
"""

ASK_ACTIVATION_INPUT_KIND = """
Choose your input data kind:
1. Encoded / hidden float tensor
2. Raw token IDs

Your choice: """

ASK_POSITION_WISE_LINEAR_INPUT_KIND = """
Choose your input data kind:
1. Encoded / hidden float tensor
2. Raw token IDs

Your choice: """

ASK_POSITION_WISE_LINEAR_PURPOSE = """
Choose the Position-wise Linear purpose:
1. Hidden projection
2. Output score projection before Softmax

Your choice: """

SOFTMAX_SELECTED = """
You selected: Softmax Layer

Softmax converts logits into probabilities.
Softmax cannot be applied directly to raw token IDs or ordinary hidden tensors.
"""

ASK_SOFTMAX_INPUT_KIND = """
Choose your input data kind:
1. Logits tensor
2. Raw token IDs
3. Encoded / hidden float tensor
4. Probabilities

Your choice: """

CONTINUE_MENU = """
Do you want to add another layer?
1. Yes
2. No
"""

CHOOSE_CONTINUE = "Choose: "
