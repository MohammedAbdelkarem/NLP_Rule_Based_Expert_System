from experta import MATCH, NOT, Rule

from facts import ComponentGroup, OutputStatus, WeightShape
from utils.printing import print_rnn_encoder_decoder_result


class RNNEncoderDecoderOutputRules:
    @Rule(
        ComponentGroup(group_index=MATCH.layer_index, group_name="rnn_encoder_decoder", group_title=MATCH.layer_title, internal_layers=MATCH.internal_layers, combine_operation=MATCH.combine_operation, input_shape=MATCH.input_shape, output_shape=MATCH.output_shape, status="computed"),
        WeightShape(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", weight_name="encoder_decoder_recurrent_parameters", shape=MATCH.parameter_shape),
        NOT(OutputStatus(layer_index=MATCH.layer_index, output_name="layer_result_printed", status="printed")),
        salience=50
    )
    def print_rnn_encoder_decoder_component_result(self, layer_index, layer_title, input_shape, output_shape, parameter_shape, internal_layers, combine_operation):
        print_rnn_encoder_decoder_result(layer_index, layer_title, input_shape, output_shape, parameter_shape, internal_layers, combine_operation)
        self.declare(OutputStatus(layer_index=layer_index, output_name="layer_result_printed", status="printed"))
