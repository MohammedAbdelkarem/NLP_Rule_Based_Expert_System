from experta import Rule, MATCH, NOT, AS, TEST

from facts import SelectedLayer, LayerParameter, OptionSelection
from prompts import (
    SOFTMAX_SELECTED,
    ASK_SOFTMAX_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_SOFTMAX_VECTOR_DIM
)


from utils.console_control import read_int_parameter, read_text_parameter
class SoftmaxParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="softmax"
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind"
        )),
        salience=75
    )
    def ask_first_softmax_input_kind(self, layer_index):
        print(SOFTMAX_SELECTED)

        choice = read_text_parameter(ASK_SOFTMAX_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_softmax_input_kind_logits(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="1",
            value="logits",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_softmax_input_kind_raw_token_ids(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="3",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_softmax_input_kind_hidden_tensor(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="3",
            value="float_hidden_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="4",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_softmax_input_kind_probabilities(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice="4",
            value="probabilities",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        TEST(lambda choice: choice != "4"),
        salience=73
    )
    def ask_softmax_input_kind_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid input kind choice. Please choose 1, 2, 3, or 4.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_SOFTMAX_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="softmax"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            value="logits",
            status="resolved"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_first_softmax_logits_tensor_shape(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        vector_dim = read_int_parameter(ASK_SOFTMAX_VECTOR_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="softmax",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="softmax",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="softmax",
            parameter_name="vector_dim",
            value=vector_dim
        ))

    @Rule(
        AS.vector_dim_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            parameter_name="vector_dim",
            value=MATCH.vector_dim
        ),
        TEST(lambda vector_dim: vector_dim <= 0),
        salience=69
    )
    def ask_softmax_vector_dim_positive_again(
        self,
        vector_dim_fact,
        layer_index,
        vector_dim
    ):
        print()
        print("Invalid Softmax vector dimension. The dimension must be a positive integer.")

        self.retract(vector_dim_fact)

        new_vector_dim = read_int_parameter(ASK_SOFTMAX_VECTOR_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="softmax",
            parameter_name="vector_dim",
            value=new_vector_dim
        ))
