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
10. Tanh Activation Layer
11. Hidden Dense + Activation Block
12. Generic Feedforward MLP Block
13. Feedforward Classifier Head
14. Embedding-to-Classifier Mean Pooling Adapter
15. Sigmoid Activation Layer
16. Binary Logistic Regression Head
17. Multinomial Logistic Regression Head
18. Binary Cross-Entropy Loss
19. Categorical Cross-Entropy Loss
20. L1/L2 Regularization Constraint
21. Audio Input Layer
22. Audio Framing + Windowing Layer
23. STFT / Spectrum Extraction ComponentGroup
24. Mel Filterbank ComponentGroup
25. Log-Mel Spectrogram Layer
26. MFCC ComponentGroup
27. Delta + Delta-Delta Feature Augmentation ComponentGroup
28. Speech Feature Normalization Layer
29. Token Embedding Layer
30. Pretrained Static Embedding Layer
31. Word-Context Co-occurrence Matrix Component
32. Count-Based Embedding Component
33. Cosine Similarity Component
34. Skip-Gram with Negative Sampling ComponentGroup
35. Embedding Visualization Component
36. Embedding Bias Analysis Component
37. Embedding Evaluation Component
38. Simple RNN Layer
39. LSTM Layer
40. Stacked RNN ComponentGroup
41. Bidirectional RNN ComponentGroup
42. RNN Encoder-Decoder ComponentGroup
43. RNN Attention Component
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

TANH_SELECTED = """
You selected: Tanh Activation Layer

Tanh is an element-wise non-linear activation.
It preserves the input tensor shape and maps values to the range [-1, 1].
"""

HIDDEN_DENSE_BLOCK_SELECTED = """
You selected: Hidden Dense + Activation Block

This is a Chapter 6 feedforward hidden block.
It applies a trainable dense projection and then a non-linear activation.
"""

FEEDFORWARD_MLP_BLOCK_SELECTED = """
You selected: Generic Feedforward MLP Block

This is a generic Chapter 6 feedforward network block.
It applies Linear -> Activation -> Linear.
"""

FEEDFORWARD_CLASSIFIER_HEAD_SELECTED = """
You selected: Feedforward Classifier Head

This is a terminal classifier head.
It expects one vector per example, not a full token sequence.
Use the pooling adapter first when the previous output is (B, T, d).
"""

EMBEDDING_POOLING_SELECTED = """
You selected: Embedding-to-Classifier Mean Pooling Adapter

This adapter converts a sequence of token embeddings into one vector per example.
It averages over the sequence axis and does not add trainable weights.
"""

ASK_FEEDFORWARD_INPUT_KIND = """
Choose the input tensor kind:
1. Feature vector tensor: (batch_size, feature_dim)
2. Sequence hidden tensor: (batch_size, sequence_length, feature_dim)
3. Raw token IDs

Your choice: """

ASK_CLASSIFIER_INPUT_KIND = """
Choose the classifier input kind:
1. Feature vector tensor: (batch_size, feature_dim)
2. Sequence hidden tensor: (batch_size, sequence_length, feature_dim)
3. Raw token IDs

Your choice: """

ASK_POOLING_INPUT_KIND = """
Choose the pooling input kind:
1. Sequence embedding tensor: (batch_size, sequence_length, embedding_dim)
2. Feature vector tensor: (batch_size, feature_dim)
3. Raw token IDs

Your choice: """

ASK_HIDDEN_DIM = "Enter hidden dimension: "
ASK_OUTPUT_DIM = "Enter output dimension: "
ASK_FEATURE_DIM = "Enter feature dimension: "
ASK_EMBEDDING_DIM = "Enter embedding dimension: "

ASK_HIDDEN_BLOCK_ACTIVATION = """
Choose the block activation function:
1. ReLU
2. Tanh

Your choice: """

ASK_MLP_BLOCK_ACTIVATION = """
Choose the MLP activation function:
1. ReLU
2. Tanh

Your choice: """

ASK_CLASSIFIER_MODE = """
Choose classifier mode:
1. Binary classification
2. Multiclass classification

Your choice: """

ASK_NUM_CLASSES = "Enter number of classes: "

# -----------------------------------------------------------------------------
# Restored prompts required by Chapter 4, Chapter 5, and speech modules.
# -----------------------------------------------------------------------------

TOKEN_EMBEDDING_SELECTED = """
You selected: Token Embedding Layer

This layer maps integer token IDs to dense vectors using a trainable embedding matrix.
"""

PRETRAINED_STATIC_EMBEDDING_SELECTED = """
You selected: Pretrained Static Embedding Layer

This layer maps token IDs to vectors loaded from a pretrained static embedding source.
The embedding matrix can be frozen or fine-tuned.
"""

COUNT_BASED_EMBEDDING_SELECTED = """
You selected: Count-Based Embedding Component

This component builds sparse word vectors from word-context count statistics.
"""

CO_OCCURRENCE_MATRIX_SELECTED = """
You selected: Word-Context Co-occurrence Matrix Component

This component represents corpus-level word-context counts.
"""

COSINE_SIMILARITY_SELECTED = """
You selected: Cosine Similarity Component

This component compares two embedding vectors without adding trainable weights.
"""

SGNS_SELECTED = """
You selected: Skip-Gram with Negative Sampling ComponentGroup

This training component learns word embeddings from target-context pairs.
"""

EMBEDDING_VISUALIZATION_SELECTED = """
You selected: Embedding Visualization Component

This analysis component projects embedding vectors for inspection.
"""

EMBEDDING_BIAS_ANALYSIS_SELECTED = """
You selected: Embedding Bias Analysis Component

This analysis component compares target and attribute embedding groups.
"""

EMBEDDING_EVALUATION_SELECTED = """
You selected: Embedding Evaluation Component

This evaluation component checks embedding quality using similarity benchmarks.
"""

ASK_VOCAB_SIZE = "Enter vocabulary size: "
ASK_WORD_VOCAB_SIZE = "Enter word vocabulary size: "
ASK_TARGET_VOCAB_SIZE = "Enter target vocabulary size: "
ASK_CONTEXT_VOCAB_SIZE = "Enter context vocabulary size: "
ASK_CONTEXT_WINDOW_SIZE = "Enter context window size: "
ASK_NEGATIVE_SAMPLE_COUNT = "Enter negative sample count: "
ASK_VECTOR_DIM = "Enter vector dimension: "
ASK_NUM_EMBEDDED_ITEMS = "Enter number of embedded items: "
ASK_PRETRAINED_EMBEDDING_SOURCE = """
Choose pretrained embedding source:
1. Word2Vec / GoogleNews-like vectors
2. GloVe-like vectors
3. FastText-like subword vectors

Your choice: """
ASK_PRETRAINED_EMBEDDING_TRAINING_MODE = """
Choose pretrained embedding training mode:
1. Frozen embedding matrix
2. Fine-tuned embedding matrix

Your choice: """
ASK_VISUALIZATION_DIM = """
Choose visualization output dimension:
1. 2D projection
2. 3D projection

Your choice: """
ASK_TARGET_WORD_COUNT = "Enter target word count: "
ASK_ATTRIBUTE_WORD_COUNT = "Enter attribute word count: "
ASK_BENCHMARK_COUNT = "Enter benchmark pair count: "
ASK_EVALUATION_PAIR_COUNT = "Enter evaluation pair count: "

SIGMOID_SELECTED = """
You selected: Sigmoid Activation Layer

Sigmoid converts one binary logit into a probability between 0 and 1.
"""

ASK_SIGMOID_INPUT_KIND = """
Choose your input data kind:
1. Binary logits with last dimension = 1
2. Raw token IDs / hidden tensor / probabilities

Your choice: """

ASK_SIGMOID_INPUT_RANK = """
Choose Sigmoid input shape rank:
1. Rank-2 binary logits: (batch_size, 1)
2. Rank-3 binary logits: (batch_size, sequence_length, 1)

Your choice: """

BINARY_LOGISTIC_REGRESSION_HEAD_SELECTED = """
You selected: Binary Logistic Regression Head

This ComponentGroup combines Linear logits projection and Sigmoid activation.
"""

MULTINOMIAL_LOGISTIC_REGRESSION_HEAD_SELECTED = """
You selected: Multinomial Logistic Regression Head

This ComponentGroup combines Linear logits projection and Softmax activation.
"""

BINARY_CROSS_ENTROPY_SELECTED = """
You selected: Binary Cross-Entropy Loss

This training component must follow binary probabilities.
"""

CATEGORICAL_CROSS_ENTROPY_SELECTED = """
You selected: Categorical Cross-Entropy Loss

This training component must follow class probabilities.
"""

REGULARIZATION_SELECTED = """
You selected: L1/L2 Regularization Constraint

This training constraint adds a penalty to trainable weights.
"""

ASK_BINARY_CLASSIFIER_INPUT_RANK = """
Choose binary classifier input shape:
1. Feature vector: (batch_size, input_dim)
2. Sequence features: (batch_size, sequence_length, input_dim)

Your choice: """

ASK_MULTICLASS_CLASSIFIER_INPUT_RANK = """
Choose multiclass classifier input shape:
1. Feature vector: (batch_size, input_dim)
2. Sequence features: (batch_size, sequence_length, input_dim)

Your choice: """

ASK_MULTICLASS_CLASS_COUNT = "Enter number of classes: "
ASK_LOSS_REDUCTION = """
Choose loss reduction:
1. Mean over batch/items
2. Sum over batch/items

Your choice: """
ASK_CCE_LABEL_FORMAT = """
Choose label format:
1. One-hot labels
2. Class-index labels

Your choice: """
ASK_REGULARIZATION_TYPE = """
Choose regularization type:
1. L1 regularization
2. L2 regularization

Your choice: """
ASK_REGULARIZATION_LAMBDA_EXPONENT = "Enter lambda exponent p for lambda = 10^-p: "

AUDIO_INPUT_SELECTED = """
You selected: Audio Input Layer

This layer declares raw audio tensor shape and sampling metadata.
"""

FRAMING_WINDOWING_SELECTED = """
You selected: Audio Framing + Windowing Layer

This layer slices audio into overlapping frames and applies a window function.
"""

STFT_SPECTRUM_SELECTED = """
You selected: STFT / Spectrum Extraction ComponentGroup

This component converts windowed frames to frequency-domain spectrum features.
"""

MEL_FILTERBANK_SELECTED = """
You selected: Mel Filterbank ComponentGroup

This component projects spectral bins onto the Mel scale.
"""

LOG_MEL_SPECTROGRAM_SELECTED = """
You selected: Log-Mel Spectrogram Layer

This layer applies log compression to Mel energies.
"""

MFCC_SELECTED = """
You selected: MFCC ComponentGroup

This component converts log-Mel features into cepstral coefficients using DCT.
"""

DELTA_FEATURES_SELECTED = """
You selected: Delta + Delta-Delta Feature Augmentation ComponentGroup

This component adds fixed temporal derivative streams to MFCC features.
"""

FEATURE_NORMALIZATION_SELECTED = """
You selected: Speech Feature Normalization Layer

This layer normalizes MFCC, Delta, and Delta-Delta streams using dataset statistics.
"""

ASK_AUDIO_BATCH_SIZE = "Enter audio batch size: "
ASK_AUDIO_DURATION_SECONDS = "Enter audio duration in seconds: "
ASK_AUDIO_SAMPLE_RATE = "Enter sample rate in Hz: "
ASK_AUDIO_CHANNELS = "Enter number of audio channels: "
ASK_AUDIO_BIT_DEPTH = "Enter audio bit depth: "
ASK_FRAME_LENGTH_MS = "Enter frame length in milliseconds: "
ASK_HOP_LENGTH_MS = "Enter hop length in milliseconds: "
ASK_WINDOW_FUNCTION = """
Choose window function:
1. Hann window
2. Hamming window
3. Rectangular window

Your choice: """
ASK_STFT_N_FFT = "Enter STFT n_fft size: "
ASK_STFT_OUTPUT_MODE = """
Choose STFT output mode:
1. Magnitude spectrum
2. Power spectrum

Your choice: """
ASK_NUM_MEL_FILTERS = "Enter number of Mel filters: "
ASK_MEL_MIN_FREQUENCY_HZ = "Enter minimum Mel frequency in Hz: "
ASK_MEL_MAX_FREQUENCY_HZ = "Enter maximum Mel frequency in Hz: "
ASK_LOG_MEL_COMPRESSION_MODE = """
Choose log-Mel compression mode:
1. Natural log
2. Log10
3. Decibel scale

Your choice: """
ASK_LOG_MEL_EPSILON = "Enter epsilon for log compression: "
ASK_NUM_MFCC_COEFFICIENTS = "Enter number of MFCC coefficients: "
ASK_MFCC_DCT_TYPE = """
Choose DCT type:
1. DCT-II
2. DCT-III

Your choice: """
ASK_MFCC_INCLUDE_C0 = """
Choose C0 handling:
1. Include C0
2. Exclude C0

Your choice: """
ASK_MFCC_LIFTER = "Enter lifter value, or 0 to disable liftering: "
ASK_DELTA_WINDOW = "Enter delta window radius: "
ASK_FEATURE_NORMALIZATION_MODE = """
Choose feature normalization mode:
1. CMVN / standardization
2. Range scaling to [-1, 1]

Your choice: """


SIMPLE_RNN_SELECTED = """
You selected: Simple RNN Layer

This Chapter 13 layer processes a sequence one time step at a time.
At each step it combines the current input vector with the previous hidden state.

Input shape:  (batch_size, sequence_length, input_dim)
Output shape: depends on return mode:
- full sequence: (batch_size, sequence_length, hidden_size)
- last hidden:   (batch_size, hidden_size)

Enter the Simple RNN configuration:
"""

LSTM_SELECTED = """
You selected: LSTM Layer

This Chapter 13 recurrent layer extends the RNN with an internal cell state and
three main gates: forget gate, input/update gate, and output gate.

Input shape:  (batch_size, sequence_length, input_dim)
Output shape: depends on return mode:
- full sequence: (batch_size, sequence_length, hidden_size)
- last hidden:   (batch_size, hidden_size)

Enter the LSTM configuration:
"""

STACKED_RNN_SELECTED = """
You selected: Stacked RNN ComponentGroup

This Chapter 13 recurrent architecture places multiple recurrent layers above each
other. Each layer receives the sequence output from the layer below it.

Input shape:  (batch_size, sequence_length, input_dim)
Output shape: depends on return mode.

Enter the stacked recurrent configuration:
"""

BIDIRECTIONAL_RNN_SELECTED = """
You selected: Bidirectional RNN ComponentGroup

This Chapter 13 recurrent architecture uses two recurrent paths:
1. Forward path reads left to right.
2. Backward path reads right to left.

The two hidden representations are concatenated.

Input shape:  (batch_size, sequence_length, input_dim)
Output shape: depends on return mode.

Enter the bidirectional recurrent configuration:
"""

RNN_ENCODER_DECODER_SELECTED = """
You selected: RNN Encoder-Decoder ComponentGroup

This Chapter 13 sequence-to-sequence architecture uses an encoder recurrent model
It uses an encoder recurrent model on the source sequence and a decoder recurrent model on the target sequence.

Encoder input shape:  (batch_size, source_sequence_length, input_dim)
Decoder output shape: (batch_size, target_sequence_length, hidden_size)

Enter the encoder-decoder configuration:
"""

RNN_ATTENTION_SELECTED = """
You selected: RNN Attention Component

This Chapter 13 component is used with encoder-decoder RNNs. It computes
attention weights over encoder states and builds context vectors for decoder states.

Attention weights shape: (batch_size, target_sequence_length, source_sequence_length)
Context output shape:   (batch_size, target_sequence_length, encoder_hidden_size)

Enter the attention configuration:
"""

ASK_HIDDEN_SIZE = "Enter recurrent hidden size: "
ASK_NUM_RECURRENT_LAYERS = "Enter number of recurrent layers: "
ASK_SOURCE_SEQUENCE_LENGTH = "Enter source sequence length: "
ASK_TARGET_SEQUENCE_LENGTH = "Enter target sequence length: "
ASK_ENCODER_HIDDEN_SIZE = "Enter encoder hidden size: "
ASK_DECODER_HIDDEN_SIZE = "Enter decoder hidden size: "
ASK_RECURRENT_RETURN_MODE = """
Choose recurrent return mode:
1. Return full sequence
2. Return last hidden state only

Your choice: """
ASK_RECURRENT_UNIT_TYPE = """
Choose recurrent unit type:
1. Simple RNN units
2. LSTM units

Your choice: """
ASK_ATTENTION_SCORING_TYPE = """
Choose RNN attention scoring type:
1. Dot-product attention
2. Bilinear attention

Your choice: """
ASK_POSITIVE_INTEGER_VALUE = "Enter a positive integer value: "


ASK_GLOBAL_ROLLBACK_LAYER = "Choose replacement layer after rollback: "
