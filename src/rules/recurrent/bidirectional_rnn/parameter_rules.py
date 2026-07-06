from experta import MATCH, NOT, Rule

from src.facts import LayerParameter, OptionSelection, SelectedLayer
from src.prompts import (
    ASK_BATCH_SIZE,
    ASK_HIDDEN_SIZE,
    ASK_INPUT_DIM,
    ASK_RECURRENT_RETURN_MODE,
    ASK_RECURRENT_UNIT_TYPE,
    ASK_SEQUENCE_LENGTH,
    BIDIRECTIONAL_RNN_SELECTED,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class BidirectionalRNNParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="bidirectional_rnn"),
        NOT(LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size")),
        salience=72
    )
    def ask_first_bidirectional_rnn_parameters(self, layer_index):
        print(BIDIRECTIONAL_RNN_SELECTED)
        unit_choice = read_text_parameter(ASK_RECURRENT_UNIT_TYPE)
        return_choice = read_text_parameter(ASK_RECURRENT_RETURN_MODE)
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        input_dim = read_int_parameter(ASK_INPUT_DIM)
        hidden_size = read_int_parameter(ASK_HIDDEN_SIZE)
        self.declare(OptionSelection(layer_index=layer_index, layer_name="bidirectional_rnn", question_key="recurrent_unit_type", choice=unit_choice, value="unknown", status="raw"))
        self.declare(OptionSelection(layer_index=layer_index, layer_name="bidirectional_rnn", question_key="recurrent_return_mode", choice=return_choice, value="unknown", status="raw"))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="bidirectional_rnn", parameter_name="batch_size", value=batch_size))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="bidirectional_rnn", parameter_name="sequence_length", value=sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="bidirectional_rnn", parameter_name="input_dim", value=input_dim))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size", value=hidden_size))

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=MATCH.previous_layer_index, layer_name="bidirectional_rnn"),
        NOT(LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size")),
        NOT(SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="bidirectional_rnn")),
        salience=72
    )
    def ask_next_bidirectional_rnn_parameters(self, layer_index):
        print(BIDIRECTIONAL_RNN_SELECTED)
        unit_choice = read_text_parameter(ASK_RECURRENT_UNIT_TYPE)
        return_choice = read_text_parameter(ASK_RECURRENT_RETURN_MODE)
        hidden_size = read_int_parameter(ASK_HIDDEN_SIZE)
        self.declare(OptionSelection(layer_index=layer_index, layer_name="bidirectional_rnn", question_key="recurrent_unit_type", choice=unit_choice, value="unknown", status="raw"))
        self.declare(OptionSelection(layer_index=layer_index, layer_name="bidirectional_rnn", question_key="recurrent_return_mode", choice=return_choice, value="unknown", status="raw"))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size", value=hidden_size))
