from experta import MATCH, NOT, Rule

from facts import Layer, LayerParameter, OptionSelection, SelectedLayer
from rules.recurrent.common import (
    declare_lstm_cell_state,
    declare_recurrent_layer,
    lstm_weight_summary,
    shape_2d,
    shape_3d,
)


class LSTMFirstCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="lstm",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="lstm",
            question_key="recurrent_return_mode",
            value="full_sequence",
            status="resolved"
        ),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="sequence_length", value=MATCH.sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="input_dim", value=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="hidden_size", value=MATCH.hidden_size),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_lstm_full_sequence(self, layer_index, layer_title, batch_size, sequence_length, input_dim, hidden_size):
        input_shape = shape_3d(batch_size, sequence_length, input_dim)
        output_shape = shape_3d(batch_size, sequence_length, hidden_size)
        parameter_shape = lstm_weight_summary(input_dim, hidden_size)
        declare_recurrent_layer(self, layer_index, "lstm", layer_title, input_shape, output_shape, 3, batch_size, sequence_length, input_dim, hidden_size, "lstm_gate_parameters", parameter_shape, "output_hidden_sequence", "float_hidden_sequence_tensor", "gated_recurrent_memory_layer", "full_sequence")
        declare_lstm_cell_state(self, layer_index, batch_size, hidden_size)

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="lstm",
            layer_title=MATCH.layer_title
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="lstm",
            question_key="recurrent_return_mode",
            value="last_hidden",
            status="resolved"
        ),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="sequence_length", value=MATCH.sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="input_dim", value=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="lstm", parameter_name="hidden_size", value=MATCH.hidden_size),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_lstm_last_hidden(self, layer_index, layer_title, batch_size, sequence_length, input_dim, hidden_size):
        input_shape = shape_3d(batch_size, sequence_length, input_dim)
        output_shape = shape_2d(batch_size, hidden_size)
        parameter_shape = lstm_weight_summary(input_dim, hidden_size)
        declare_recurrent_layer(self, layer_index, "lstm", layer_title, input_shape, output_shape, 2, batch_size, sequence_length, input_dim, hidden_size, "lstm_gate_parameters", parameter_shape, "output_last_hidden_state", "float_hidden_state_tensor", "gated_recurrent_memory_layer", "last_hidden")
        declare_lstm_cell_state(self, layer_index, batch_size, hidden_size)
