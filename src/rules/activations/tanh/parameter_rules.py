from experta import AS, MATCH, NOT, Rule, TEST

from src.facts import LayerParameter, OptionSelection, SelectedLayer
from src.prompts import (
    ASK_ACTIVATION_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_MODEL_DIM,
    ASK_SEQUENCE_LENGTH,
    TANH_SELECTED,
)


class TanhParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="tanh",
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
        )),
        salience=75,
    )
    def ask_first_tanh_input_kind(self, layer_index):
        print(TANH_SELECTED)
        choice = input(ASK_ACTIVATION_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_tanh_input_kind_hidden_tensor(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice="1",
            value="encoded_float_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_tanh_input_kind_raw_token_ids(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73,
    )
    def ask_tanh_input_kind_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid input kind choice. Please choose 1 or 2.")
        self.retract(option_fact)
        new_choice = input(ASK_ACTIVATION_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="tanh",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            question_key="activation_input_data_kind",
            value="encoded_float_tensor",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="tanh",
            parameter_name="batch_size",
        )),
        salience=70,
    )
    def ask_first_tanh_parameters(self, layer_index):
        batch_size = int(input(ASK_BATCH_SIZE))
        sequence_length = int(input(ASK_SEQUENCE_LENGTH))
        model_dim = int(input(ASK_MODEL_DIM))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="tanh",
            parameter_name="batch_size",
            value=batch_size,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="tanh",
            parameter_name="sequence_length",
            value=sequence_length,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="tanh",
            parameter_name="model_dim",
            value=model_dim,
        ))
