from experta import MATCH, NOT, Rule

from facts import Layer, OutputStatus, WeightShape
from utils.printing import print_simple_rnn_result


class SimpleRNNOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="simple_rnn",
            weight_name="simple_rnn_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_simple_rnn_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape
    ):
        print_simple_rnn_result(
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape
        )
        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
