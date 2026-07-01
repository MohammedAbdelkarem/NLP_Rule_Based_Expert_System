from experta import Rule, MATCH, NOT

from src.facts import (
    Layer,
    LayerParameter,
    WeightShape,
    OutputStatus
)
from src.utils.print_helpers import print_residual_connection_result


class ResidualConnectionOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            weight_name="residual_connection_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_residual_connection_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        source_layer_index,
        parameter_shape
    ):
        print_residual_connection_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="residual_connection",
            layer_kind="merge_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            source_layer_index=source_layer_index,
            parameter_shape=parameter_shape
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
