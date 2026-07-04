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
10. Audio Input Layer (Speech Module)
11. Audio Framing + Windowing Layer (Speech Module)
12. STFT / Spectrum Extraction Component Group (Speech Module)
13. Mel Filterbank Component Group (Speech Module)
14. Log-Mel Spectrogram Layer (Speech Module)
15. MFCC Component Group (Speech Module)
16. Delta + Delta-Delta Feature Augmentation Component Group (Speech Module)
17. Speech Feature Normalization Layer (Speech Module)
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


AUDIO_INPUT_SELECTED = """
You selected: Audio Input Layer (Speech Module)

This is the first speech module layer.
It represents a WAV/audio signal as a numeric waveform tensor.
The channel dimension is preserved, so the user can choose mono, stereo, or more channels.
"""

ASK_AUDIO_BATCH_SIZE = "Enter audio batch size: "
ASK_AUDIO_DURATION_SECONDS = "Enter duration in seconds (integer): "
ASK_AUDIO_SAMPLE_RATE = "Enter sample rate in Hz: "
ASK_AUDIO_CHANNELS = "Enter number of audio channels: "
ASK_AUDIO_BIT_DEPTH = "Enter bit depth: "


FRAMING_WINDOWING_SELECTED = """
You selected: Audio Framing + Windowing Layer (Speech Module)

This layer splits the waveform into short overlapping frames and applies a window function.
It keeps the channel dimension, so mono, stereo, and multi-channel audio remain supported.
Typical speech settings are 25 ms frame length and 10 ms hop length.
"""

ASK_FRAME_LENGTH_MS = "Enter frame length in milliseconds: "
ASK_HOP_LENGTH_MS = "Enter hop length / frame shift in milliseconds: "
ASK_WINDOW_FUNCTION = """
Choose window function:
1. Hann
2. Hamming
3. Rectangular

Your choice: """


STFT_SPECTRUM_SELECTED = """
You selected: STFT / Spectrum Extraction Component Group (Speech Module)

This component group converts each windowed audio frame from the time domain to the frequency domain.
It keeps the frame and channel dimensions, so mono, stereo, and multi-channel audio remain supported.
It has no trainable weights; internally it uses fixed signal-processing components configured by n_fft.
"""

ASK_STFT_N_FFT = "Enter n_fft / FFT size: "
ASK_STFT_OUTPUT_MODE = """
Choose STFT output mode:
1. Magnitude spectrum
2. Power spectrum
3. Complex spectrum

Your choice: """


MEL_FILTERBANK_SELECTED = """
You selected: Mel Filterbank Component Group (Speech Module)

This component group converts the linear frequency spectrum from STFT into Mel frequency bands.
It preserves the frame and channel dimensions, so mono, stereo, and multi-channel audio remain supported.
It has a fixed Mel filterbank matrix, but this matrix is not trainable.
"""

ASK_NUM_MEL_FILTERS = "Enter number of Mel filters / Mel bands: "
ASK_MEL_MIN_FREQUENCY_HZ = "Enter minimum Mel frequency in Hz: "
ASK_MEL_MAX_FREQUENCY_HZ = "Enter maximum Mel frequency in Hz: "



LOG_MEL_SPECTROGRAM_SELECTED = """
You selected: Log-Mel Spectrogram Layer (Speech Module)

This layer applies element-wise log compression to Mel energy features.
It preserves the shape and channel dimension, so mono, stereo, and multi-channel audio remain supported.
It has no trainable weights, no bias vector, and no matrix multiplication.
"""

ASK_LOG_MEL_EPSILON = "Enter epsilon exponent p so epsilon = 10^-p. For 1e-6, enter 6: "
ASK_LOG_MEL_COMPRESSION_MODE = """
Choose Log-Mel compression mode:
1. Natural log: log(mel_energy + epsilon)
2. Base-10 log: log10(mel_energy + epsilon)
3. Decibel: 10 * log10(mel_energy + epsilon)

Your choice: """


MFCC_SELECTED = """
You selected: MFCC Component Group (Speech Module)

This component group converts a Log-Mel spectrogram into Mel-Frequency Cepstral Coefficients.
It uses a fixed DCT matrix over the Mel axis, so it has a matrix but no trainable weights.
It preserves the batch, frame, and channel dimensions while reducing the feature dimension.
"""

ASK_NUM_MFCC_COEFFICIENTS = "Enter number of MFCC coefficients: "
ASK_MFCC_DCT_TYPE = """
Choose MFCC DCT type:
1. DCT-II (standard MFCC choice)
2. DCT-III

Your choice: """
ASK_MFCC_INCLUDE_C0 = """
Include c0 coefficient?
1. Yes, include c0
2. No, start from c1

Your choice: """
ASK_MFCC_LIFTER = "Enter cepstral lifter value (0 to disable): "


DELTA_FEATURES_SELECTED = """
You selected: Delta + Delta-Delta Feature Augmentation Component Group (Speech Module)

This component group augments MFCC features with fixed temporal-change features.
It keeps a static MFCC stream, computes Delta features, computes Delta-Delta features,
and includes a built-in concatenation adapter that can merge the streams on the feature axis.
The temporal kernels are fixed, so there are no trainable weights.
"""

ASK_DELTA_WINDOW = "Enter delta window radius N in frames (typical value: 2): "


FEATURE_NORMALIZATION_SELECTED = """
You selected: Speech Feature Normalization Layer (Speech Module)

This layer normalizes the separated MFCC, Delta, and Delta-Delta streams.
It preserves the separated-stream design and does not force concatenation.
It uses stored training-dataset statistics, not trainable weights.
A concatenation adapter remains available later if the next layer requires one tensor.
"""

ASK_FEATURE_NORMALIZATION_MODE = """
Choose feature normalization mode:
1. CMVN / standardization: (x - mean) / (std + epsilon)
2. Range scaling: scale features to [-1, 1]

Your choice: """
