from experta import MATCH, NOT, Rule

from src.facts import ComponentGroup, Layer, LayerParameter, OptionSelection, SelectedLayer, TensorInfo
from src.rules.recurrent.common import (
    declare_recurrent_component_group,
    shape_2d,
    shape_3d,
    stacked_rnn_weight_summary,
)


class StackedRNNPreviousCalculationRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=MATCH.previous_layer_index, layer_name="stacked_rnn", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="stacked_rnn", question_key="recurrent_unit_type", value=MATCH.unit_type, status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="stacked_rnn", question_key="recurrent_return_mode", value="full_sequence", status="resolved"),
        Layer(layer_index=MATCH.previous_layer_index, output_shape=MATCH.input_shape),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_batch_size", shape=MATCH.batch_size),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_sequence_length", shape=MATCH.sequence_length),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_model_dim", shape=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="stacked_rnn", parameter_name="hidden_size", value=MATCH.hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="stacked_rnn", parameter_name="num_layers", value=MATCH.num_layers),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_next_stacked_rnn_full_sequence(self, layer_index, layer_title, input_shape, batch_size, sequence_length, input_dim, hidden_size, num_layers, unit_type):
        input_shape = input_shape
        output_shape = shape_3d(batch_size, sequence_length, hidden_size)
        parameter_shape = stacked_rnn_weight_summary(input_dim, hidden_size, num_layers, unit_type)
        declare_recurrent_component_group(self, layer_index, "stacked_rnn", layer_title, input_shape, output_shape, 3, batch_size, sequence_length, input_dim, hidden_size, hidden_size, parameter_shape, "multiple recurrent layers", "upper layer consumes lower layer hidden sequence", "deep_recurrent_sequence_encoder", "full_sequence")
        self.declare(Layer(layer_index=layer_index, layer_name="stacked_rnn", layer_title=layer_title, input_shape=input_shape, output_shape=output_shape))

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=MATCH.previous_layer_index, layer_name="stacked_rnn", layer_title=MATCH.layer_title),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="stacked_rnn", question_key="recurrent_unit_type", value=MATCH.unit_type, status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="stacked_rnn", question_key="recurrent_return_mode", value="last_hidden", status="resolved"),
        Layer(layer_index=MATCH.previous_layer_index, output_shape=MATCH.input_shape),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_batch_size", shape=MATCH.batch_size),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_sequence_length", shape=MATCH.sequence_length),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_model_dim", shape=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="stacked_rnn", parameter_name="hidden_size", value=MATCH.hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="stacked_rnn", parameter_name="num_layers", value=MATCH.num_layers),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_next_stacked_rnn_last_hidden(self, layer_index, layer_title, input_shape, batch_size, sequence_length, input_dim, hidden_size, num_layers, unit_type):
        input_shape = input_shape
        output_shape = shape_2d(batch_size, hidden_size)
        parameter_shape = stacked_rnn_weight_summary(input_dim, hidden_size, num_layers, unit_type)
        declare_recurrent_component_group(self, layer_index, "stacked_rnn", layer_title, input_shape, output_shape, 2, batch_size, sequence_length, input_dim, hidden_size, hidden_size, parameter_shape, "multiple recurrent layers", "upper layer consumes lower layer hidden sequence", "deep_recurrent_sequence_encoder", "last_hidden")
        self.declare(Layer(layer_index=layer_index, layer_name="stacked_rnn", layer_title=layer_title, input_shape=input_shape, output_shape=output_shape))
