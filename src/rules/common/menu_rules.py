from experta import Rule, MATCH, TEST, NOT, AS

from src.facts import MenuChoice, SelectedLayer, SessionStatus
from src.prompts import MAIN_MENU, ASK_ROLLBACK_LAYER


class MenuRules:
    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="1"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_input_encoding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="input_encoding",
            layer_title="Input Encoding Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="2"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_layer_norm(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="layer_norm",
            layer_title="LayerNorm Layer",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="3"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_multi_head_attention(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            layer_title="Multi-Head Attention Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="4"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_residual_connection(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="residual_connection",
            layer_title="Residual Connection Layer",
            source_name="transformers"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="5"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_feed_forward(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            layer_title="Feed-Forward Network Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="6"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_relu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="relu",
            layer_title="ReLU Activation Layer",
            source_name="activations"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="7"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_gelu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="gelu",
            layer_title="GELU Activation Layer",
            source_name="activations"
        ))


    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="8"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_position_wise_linear(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            layer_title="Position-wise Linear Layer",
            source_name="projections"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="9"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_softmax(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            layer_title="Softmax Layer",
            source_name="activations"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="10"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_audio_input(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="audio_input",
            layer_title="Audio Input Layer (Speech Module)",
            source_name="speech"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="11"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_framing_windowing(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            layer_title="Audio Framing + Windowing Layer (Speech Module)",
            source_name="speech"
        ))




    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="12"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_stft_spectrum(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="stft_spectrum",
            layer_title="STFT / Spectrum Extraction Component Group (Speech Module)",
            source_name="speech"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="13"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_mel_filterbank(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mel_filterbank",
            layer_title="Mel Filterbank Component Group (Speech Module)",
            source_name="speech"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="14"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_log_mel_spectrogram(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrogram",
            layer_title="Log-Mel Spectrogram Layer (Speech Module)",
            source_name="speech"
        ))


    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="15"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_mfcc(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mfcc",
            layer_title="MFCC Component Group (Speech Module)",
            source_name="speech"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="16"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_delta_features(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="delta_features",
            layer_title="Delta + Delta-Delta Feature Augmentation Component Group (Speech Module)",
            source_name="speech"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="17"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_feature_normalization(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feature_normalization",
            layer_title="Speech Feature Normalization Layer (Speech Module)",
            source_name="speech"
        ))







    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="18",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_sigmoid(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="sigmoid",
            layer_title="Sigmoid Activation Layer",
            source_name="activations"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="19",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_binary_logistic_regression_head(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="binary_logistic_regression_head",
            layer_title="Binary Logistic Regression Head (Chapter 4)",
            source_name="classification_heads"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="20",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_multinomial_logistic_regression_head(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multinomial_logistic_regression_head",
            layer_title="Multinomial Logistic Regression Head (Chapter 4)",
            source_name="classification_heads"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="21",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_binary_cross_entropy(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="binary_cross_entropy",
            layer_title="Binary Cross-Entropy Loss (Training Component)",
            source_name="losses"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="22",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_categorical_cross_entropy(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="categorical_cross_entropy",
            layer_title="Categorical Cross-Entropy Loss (Training Component)",
            source_name="losses"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="23",
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_l1_l2_regularization(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="l1_l2_regularization",
            layer_title="L1/L2 Regularization Constraint (Training Component)",
            source_name="regularization"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="0"
        ),
        NOT(SessionStatus(status="ended")),
        salience=90
    )
    def end_session_from_layer_menu(self, layer_index, previous_layer_index):
        self.declare(SessionStatus(
            status="ended",
            reason="user_stopped_from_layer_menu"
        ))

    @Rule(
        AS.menu_choice_fact << MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice=MATCH.choice
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
        TEST(lambda choice: choice != "0"),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=89
    )
    def ask_layer_choice_again(
        self,
        menu_choice_fact,
        layer_index,
        previous_layer_index,
        choice
    ):
        print()
        print("Invalid layer choice. Please choose a listed option number or 0.")
        print(MAIN_MENU)

        self.retract(menu_choice_fact)

        new_choice = input(ASK_ROLLBACK_LAYER)

        self.declare(MenuChoice(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            choice=new_choice
        ))
