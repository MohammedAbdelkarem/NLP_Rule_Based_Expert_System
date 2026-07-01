from experta import Rule, MATCH, NOT

from src.facts import Layer, WeightShape, OutputStatus
from src.utils.print_helpers import print_simple_layer_result


class LayerNormOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            weight_name="layer_norm_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_layer_norm_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        parameter_shape,
        output_shape
    ):
        print_simple_layer_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="layer_norm",
            layer_kind="simple_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            data_type="float hidden tensor"
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
