from experta import AS, MATCH, NOT, Rule, TEST

from facts import LayerParameter, OptionSelection, SelectedLayer, TensorInfo
from prompts import (
    ASK_BATCH_SIZE,
    ASK_FEATURE_DIM,
    ASK_FEEDFORWARD_INPUT_KIND,
    ASK_HIDDEN_BLOCK_ACTIVATION,
    ASK_HIDDEN_DIM,
    ASK_SEQUENCE_LENGTH,
    HIDDEN_DENSE_BLOCK_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class HiddenDenseActivationBlockParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="hidden_dense_activation_block",
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
        )),
        salience=75,
    )
    def ask_first_hidden_dense_input_kind(self, layer_index):
        print(HIDDEN_DENSE_BLOCK_SELECTED)
        choice = read_text_parameter(ASK_FEEDFORWARD_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_hidden_dense_feature_vector_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="1",
            value="feature_vector_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_hidden_dense_sequence_tensor_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="2",
            value="sequence_hidden_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="3",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_hidden_dense_raw_token_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice="3",
            value="raw_token_ids",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=73,
    )
    def ask_hidden_dense_input_kind_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid input kind choice. Please choose 1, 2, or 3.")
        self.retract(option_fact)
        new_choice = read_text_parameter(ASK_FEEDFORWARD_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="hidden_dense_activation_block",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            value="feature_vector_tensor",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="batch_size",
        )),
        salience=70,
    )
    def ask_first_hidden_dense_rank_2_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        input_dim = read_int_parameter(ASK_FEATURE_DIM)
        hidden_dim = read_int_parameter(ASK_HIDDEN_DIM)
        activation_choice = read_text_parameter(ASK_HIDDEN_BLOCK_ACTIVATION)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="batch_size",
            value=batch_size,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="model_dim",
            value=input_dim,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="hidden_dim",
            value=hidden_dim,
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice=activation_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="hidden_dense_activation_block",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="feedforward_input_kind",
            value="sequence_hidden_tensor",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="batch_size",
        )),
        salience=70,
    )
    def ask_first_hidden_dense_rank_3_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        input_dim = read_int_parameter(ASK_FEATURE_DIM)
        hidden_dim = read_int_parameter(ASK_HIDDEN_DIM)
        activation_choice = read_text_parameter(ASK_HIDDEN_BLOCK_ACTIVATION)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="batch_size",
            value=batch_size,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="sequence_length",
            value=sequence_length,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="model_dim",
            value=input_dim,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="hidden_dim",
            value=hidden_dim,
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice=activation_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="hidden_dense_activation_block",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            data_type="float_hidden_tensor",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="hidden_dim",
        )),
        salience=70,
    )
    def ask_hidden_dense_after_previous_parameters(self, layer_index):
        hidden_dim = read_int_parameter(ASK_HIDDEN_DIM)
        activation_choice = read_text_parameter(ASK_HIDDEN_BLOCK_ACTIVATION)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            parameter_name="hidden_dim",
            value=hidden_dim,
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice=activation_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=69,
    )
    def resolve_hidden_dense_relu_activation(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice="1",
            value="relu",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=69,
    )
    def resolve_hidden_dense_tanh_activation(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice="2",
            value="tanh",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=68,
    )
    def ask_hidden_dense_activation_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid activation choice. Please choose 1 or 2.")
        self.retract(option_fact)
        new_choice = read_text_parameter(ASK_HIDDEN_BLOCK_ACTIVATION)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="hidden_dense_activation_block",
            question_key="hidden_block_activation",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))
