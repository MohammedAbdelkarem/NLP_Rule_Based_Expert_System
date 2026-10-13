from experta import Rule, MATCH, NOT

from facts import Layer, WeightShape, OutputStatus
from utils.print_helpers import print_activation_layer_result


class GELUOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="gelu",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="gelu",
            weight_name="activation_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_gelu_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape
    ):
        print_activation_layer_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="gelu",
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
