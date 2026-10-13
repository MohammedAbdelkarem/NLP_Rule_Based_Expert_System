from experta import MATCH, NOT, Rule

from facts import ComponentGroup, Layer, LayerParameter, OptionSelection, SelectedLayer
from rules.recurrent.common import (
    bidirectional_weight_summary,
    declare_recurrent_component_group,
    doubled,
    shape_2d,
    shape_3d,
)


class BidirectionalRNNFirstCalculationRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="bidirectional_rnn", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", question_key="recurrent_unit_type", value=MATCH.unit_type, status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", question_key="recurrent_return_mode", value="full_sequence", status="resolved"),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="sequence_length", value=MATCH.sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="input_dim", value=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size", value=MATCH.hidden_size),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_bidirectional_rnn_full_sequence(self, layer_index, layer_title, batch_size, sequence_length, input_dim, hidden_size, unit_type):
        input_shape = shape_3d(batch_size, sequence_length, input_dim)
        output_dim = doubled(hidden_size)
        output_shape = shape_3d(batch_size, sequence_length, output_dim)
        parameter_shape = bidirectional_weight_summary(input_dim, hidden_size, unit_type)
        declare_recurrent_component_group(self, layer_index, "bidirectional_rnn", layer_title, input_shape, output_shape, 3, batch_size, sequence_length, input_dim, output_dim, hidden_size, parameter_shape, "forward recurrent path + backward recurrent path", "concatenate forward and backward hidden states", "bidirectional_context_encoder", "full_sequence")
        self.declare(Layer(layer_index=layer_index, layer_name="bidirectional_rnn", layer_title=layer_title, input_shape=input_shape, output_shape=output_shape))

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="bidirectional_rnn", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", question_key="recurrent_unit_type", value=MATCH.unit_type, status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", question_key="recurrent_return_mode", value="last_hidden", status="resolved"),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="sequence_length", value=MATCH.sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="input_dim", value=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="bidirectional_rnn", parameter_name="hidden_size", value=MATCH.hidden_size),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_bidirectional_rnn_last_hidden(self, layer_index, layer_title, batch_size, sequence_length, input_dim, hidden_size, unit_type):
        input_shape = shape_3d(batch_size, sequence_length, input_dim)
        output_dim = doubled(hidden_size)
        output_shape = shape_2d(batch_size, output_dim)
        parameter_shape = bidirectional_weight_summary(input_dim, hidden_size, unit_type)
        declare_recurrent_component_group(self, layer_index, "bidirectional_rnn", layer_title, input_shape, output_shape, 2, batch_size, sequence_length, input_dim, output_dim, hidden_size, parameter_shape, "forward recurrent path + backward recurrent path", "concatenate forward and backward hidden states", "bidirectional_context_encoder", "last_hidden")
        self.declare(Layer(layer_index=layer_index, layer_name="bidirectional_rnn", layer_title=layer_title, input_shape=input_shape, output_shape=output_shape))
