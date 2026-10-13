from experta import MATCH, NOT, Rule

from facts import ComponentGroup, OutputStatus, TensorInfo, WeightShape
from utils.printing import print_rnn_attention_result


class RNNAttentionOutputRules:
    @Rule(
        ComponentGroup(group_index=MATCH.layer_index, group_name="rnn_attention", group_title=MATCH.layer_title, internal_layers=MATCH.internal_layers, combine_operation=MATCH.combine_operation, input_shape=MATCH.input_shape, output_shape=MATCH.output_shape, status="computed"),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="rnn_attention", tensor_name="attention_weights", shape=MATCH.attention_weights_shape),
        WeightShape(layer_index=MATCH.layer_index, layer_name="rnn_attention", weight_name="attention_scoring_parameters", shape=MATCH.parameter_shape),
        NOT(OutputStatus(layer_index=MATCH.layer_index, output_name="layer_result_printed", status="printed")),
        salience=50
    )
    def print_rnn_attention_component_result(self, layer_index, layer_title, input_shape, output_shape, attention_weights_shape, parameter_shape, internal_layers, combine_operation):
        print_rnn_attention_result(layer_index, layer_title, input_shape, output_shape, attention_weights_shape, parameter_shape, internal_layers, combine_operation)
        self.declare(OutputStatus(layer_index=layer_index, output_name="layer_result_printed", status="printed"))
