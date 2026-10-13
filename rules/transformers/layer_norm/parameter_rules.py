from experta import Rule, MATCH, NOT, AS, TEST

from facts import SelectedLayer, LayerParameter, OptionSelection
from prompts import (
    LAYER_NORM_SELECTED,
    ASK_LAYER_NORM_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_MODEL_DIM
)


from utils.console_control import read_int_parameter, read_text_parameter
class LayerNormParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="layer_norm"
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind"
        )),
        salience=75
    )
    def ask_first_layer_norm_input_kind(self, layer_index):
        print(LAYER_NORM_SELECTED)

        choice = read_text_parameter(ASK_LAYER_NORM_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_layer_norm_input_kind_hidden_tensor(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice="1",
            value="encoded_float_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_layer_norm_input_kind_raw_token_ids(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_layer_norm_input_kind_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid input kind choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_LAYER_NORM_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="layer_norm"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            value="encoded_float_tensor",
            status="resolved"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_first_layer_norm_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        model_dim = read_int_parameter(ASK_MODEL_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="layer_norm",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="layer_norm",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="layer_norm",
            parameter_name="model_dim",
            value=model_dim
        ))
