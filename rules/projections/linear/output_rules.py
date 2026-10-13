from experta import Rule, MATCH, NOT

from facts import (
    Layer,
    WeightShape,
    TensorInfo,
    OutputStatus
)
from utils.print_helpers import print_position_wise_linear_result


class PositionWiseLinearOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            weight_name="linear_parameters",
            shape=MATCH.parameter_shape
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            role="output",
            data_type=MATCH.output_data_type
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_position_wise_linear_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape,
        output_data_type
    ):
        print_position_wise_linear_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="position_wise_linear",
            layer_kind="simple_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            output_data_type=output_data_type
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
