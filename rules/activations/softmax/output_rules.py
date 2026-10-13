from experta import Rule, MATCH, NOT

from facts import Layer, WeightShape, OutputStatus
from utils.print_helpers import print_softmax_result


class SoftmaxOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            weight_name="softmax_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_softmax_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape
    ):
        print_softmax_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="softmax",
            layer_kind="simple_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
