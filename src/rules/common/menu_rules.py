from experta import Rule, MATCH, TEST, NOT, AS

from src.facts import MenuChoice, SelectedLayer, SessionStatus
from src.prompts import MAIN_MENU, ASK_ROLLBACK_LAYER


from src.utils.console_control import read_int_parameter, read_text_parameter
class MenuRules:
    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="1",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_input_encoding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="input_encoding",
            layer_title="Input Encoding (Embedding + Positional Embedding + Summation)",
            source_name="transformers",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="2",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_layer_norm(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="layer_norm",
            layer_title="LayerNorm Layer",
            source_name="transformers",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="3",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_multi_head_attention(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            layer_title="Multi-Head Attention Layer Group",
            source_name="transformers",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="4",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_residual_connection(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="residual_connection",
            layer_title="Residual Connection Layer",
            source_name="transformers",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="5",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_feed_forward(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            layer_title="Feed-Forward Network Layer Group",
            source_name="transformers",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="6",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_relu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="relu",
            layer_title="ReLU Activation Layer",
            source_name="activations",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="7",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_gelu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="gelu",
            layer_title="GELU Activation Layer",
            source_name="activations",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="8",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_position_wise_linear(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            layer_title="Position-wise Linear Layer",
            source_name="projections",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="9",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_softmax(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            layer_title="Softmax Layer",
            source_name="activations",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="10",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_tanh(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="tanh",
            layer_title="Tanh Activation Layer",
            source_name="activations",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="11",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_hidden_dense_activation_block(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="hidden_dense_activation_block",
            layer_title="Hidden Dense + Activation Block",
            source_name="feedforward",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="12",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_feedforward_mlp_block(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_mlp_block",
            layer_title="Generic Feedforward MLP Block",
            source_name="feedforward",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="13",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_feedforward_classifier_head(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_classifier_head",
            layer_title="Feedforward Classifier Head",
            source_name="classification_heads",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="14",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_embedding_to_classifier_pooling(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
            layer_title="Embedding-to-Classifier Mean Pooling Adapter",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="15",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_sigmoid(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="sigmoid",
            layer_title="Sigmoid Activation Layer",
            source_name="activations",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="16",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_binary_logistic_regression_head(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="binary_logistic_regression_head",
            layer_title="Binary Logistic Regression Head",
            source_name="classification_heads",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="17",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_multinomial_logistic_regression_head(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multinomial_logistic_regression_head",
            layer_title="Multinomial Logistic Regression Head",
            source_name="classification_heads",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="18",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_binary_cross_entropy(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="binary_cross_entropy",
            layer_title="Binary Cross-Entropy Loss",
            source_name="losses",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="19",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_categorical_cross_entropy(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title="Categorical Cross-Entropy Loss",
            source_name="losses",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="20",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_l1_l2_regularization(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="l1_l2_regularization",
            layer_title="L1/L2 Regularization Constraint",
            source_name="regularization",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="21",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_audio_input(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="audio_input",
            layer_title="Audio Input Layer",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="22",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_framing_windowing(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            layer_title="Audio Framing + Windowing Layer",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="23",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_stft_spectrum(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="stft_spectrum",
            layer_title="STFT / Spectrum Extraction ComponentGroup",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="24",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_mel_filterbank(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mel_filterbank",
            layer_title="Mel Filterbank ComponentGroup",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="25",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_log_mel_spectrogram(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrogram",
            layer_title="Log-Mel Spectrogram Layer",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="26",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_mfcc(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mfcc",
            layer_title="MFCC ComponentGroup",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="27",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_delta_features(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="delta_features",
            layer_title="Delta + Delta-Delta Feature Augmentation ComponentGroup",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="28",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_feature_normalization(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feature_normalization",
            layer_title="Speech Feature Normalization Layer",
            source_name="speech",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="29",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_token_embedding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="token_embedding",
            layer_title="Token Embedding Layer",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="30",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_pretrained_static_embedding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="pretrained_static_embedding",
            layer_title="Pretrained Static Embedding Layer",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="31",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_co_occurrence_matrix(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="co_occurrence_matrix",
            layer_title="Word-Context Co-occurrence Matrix Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="32",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_count_based_embedding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="count_based_embedding",
            layer_title="Count-Based Embedding Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="33",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_cosine_similarity(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="cosine_similarity",
            layer_title="Cosine Similarity Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="34",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_skipgram_sgns(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="skipgram_sgns",
            layer_title="Skip-Gram with Negative Sampling ComponentGroup",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="35",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_embedding_visualization(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_visualization",
            layer_title="Embedding Visualization Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="36",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_embedding_bias_analysis(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_bias_analysis",
            layer_title="Embedding Bias Analysis Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="37",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_embedding_evaluation(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_evaluation",
            layer_title="Embedding Evaluation Component",
            source_name="embeddings",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="38",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_simple_rnn(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="simple_rnn",
            layer_title="Simple RNN Layer",
            source_name="recurrent",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="39",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_lstm(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="lstm",
            layer_title="LSTM Layer",
            source_name="recurrent",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="40",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_stacked_rnn(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="stacked_rnn",
            layer_title="Stacked RNN ComponentGroup",
            source_name="recurrent",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="41",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_bidirectional_rnn(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="bidirectional_rnn",
            layer_title="Bidirectional RNN ComponentGroup",
            source_name="recurrent",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="42",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_rnn_encoder_decoder(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="rnn_encoder_decoder",
            layer_title="RNN Encoder-Decoder ComponentGroup",
            source_name="sequence_to_sequence",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="43",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90,
    )
    def select_rnn_attention(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="rnn_attention",
            layer_title="RNN Attention Component",
            source_name="attention",
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="0",
        ),
        NOT(SessionStatus(status="ended")),
        salience=90,
    )
    def end_session_from_layer_menu(self, layer_index, previous_layer_index):
        self.declare(SessionStatus(
            status="ended",
            reason="user_stopped_from_layer_menu",
        ))

    @Rule(
        AS.menu_choice_fact << MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice=MATCH.choice,
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        TEST(lambda choice: choice != "4"),
        TEST(lambda choice: choice != "5"),
        TEST(lambda choice: choice != "6"),
        TEST(lambda choice: choice != "7"),
        TEST(lambda choice: choice != "8"),
        TEST(lambda choice: choice != "9"),
        TEST(lambda choice: choice != "10"),
        TEST(lambda choice: choice != "11"),
        TEST(lambda choice: choice != "12"),
        TEST(lambda choice: choice != "13"),
        TEST(lambda choice: choice != "14"),
        TEST(lambda choice: choice != "15"),
        TEST(lambda choice: choice != "16"),
        TEST(lambda choice: choice != "17"),
        TEST(lambda choice: choice != "18"),
        TEST(lambda choice: choice != "19"),
        TEST(lambda choice: choice != "20"),
        TEST(lambda choice: choice != "21"),
        TEST(lambda choice: choice != "22"),
        TEST(lambda choice: choice != "23"),
        TEST(lambda choice: choice != "24"),
        TEST(lambda choice: choice != "25"),
        TEST(lambda choice: choice != "26"),
        TEST(lambda choice: choice != "27"),
        TEST(lambda choice: choice != "28"),
        TEST(lambda choice: choice != "29"),
        TEST(lambda choice: choice != "30"),
        TEST(lambda choice: choice != "31"),
        TEST(lambda choice: choice != "32"),
        TEST(lambda choice: choice != "33"),
        TEST(lambda choice: choice != "34"),
        TEST(lambda choice: choice != "35"),
        TEST(lambda choice: choice != "36"),
        TEST(lambda choice: choice != "37"),
        TEST(lambda choice: choice != "38"),
        TEST(lambda choice: choice != "39"),
        TEST(lambda choice: choice != "40"),
        TEST(lambda choice: choice != "41"),
        TEST(lambda choice: choice != "42"),
        TEST(lambda choice: choice != "43"),
        TEST(lambda choice: choice != "0"),
        NOT(SessionStatus(status="ended")),
        salience=89,
    )
    def ask_layer_choice_again(
        self,
        menu_choice_fact,
        layer_index,
        previous_layer_index,
        choice,
    ):
        print()
        print("Invalid layer choice. Please choose a listed option number or 0.")
        print(MAIN_MENU)

        self.retract(menu_choice_fact)

        new_choice = read_text_parameter(ASK_ROLLBACK_LAYER)

        self.declare(MenuChoice(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            choice=new_choice,
        ))
