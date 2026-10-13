from experta import MATCH, NOT, Rule

from facts import Layer, OutputStatus, WeightShape
from utils.printing import print_lstm_result


class LSTMOutputRules:
    @Rule(
        Layer(layer_index=MATCH.layer_index, layer_name="lstm", layer_title=MATCH.layer_title, input_shape=MATCH.input_shape, output_shape=MATCH.output_shape),
        WeightShape(layer_index=MATCH.layer_index, layer_name="lstm", weight_name="lstm_gate_parameters", shape=MATCH.parameter_shape),
        NOT(OutputStatus(layer_index=MATCH.layer_index, output_name="layer_result_printed", status="printed")),
        salience=50
    )
    def print_lstm_layer_result(self, layer_index, layer_title, input_shape, output_shape, parameter_shape):
        print_lstm_result(layer_index, layer_title, input_shape, output_shape, parameter_shape)
        self.declare(OutputStatus(layer_index=layer_index, output_name="layer_result_printed", status="printed"))
