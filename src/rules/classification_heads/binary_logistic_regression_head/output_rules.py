from experta import MATCH, NOT, Rule

from src.facts import ComponentGroup, OutputStatus, WeightShape
from src.utils.print_helpers import print_logistic_regression_head_result


class BinaryLogisticRegressionHeadOutputRules:
    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="binary_logistic_regression_head",
            group_title=MATCH.layer_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed",
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="binary_logistic_regression_head",
            weight_name="classifier_parameters",
            shape=MATCH.parameter_shape,
        ),
        NOT(
            OutputStatus(
                layer_index=MATCH.layer_index,
                output_name="layer_result_printed",
                status="printed",
            ),
        ),
        salience=50,
    )
    def print_binary_head_result(
        self,
        layer_index,
        layer_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        parameter_shape
    ):
        print_logistic_regression_head_result(
            layer_index,
            layer_title,
            "binary_logistic_regression_head",
            "component_group",
            internal_layers,
            combine_operation,
            input_shape,
            output_shape,
            parameter_shape,
            "binary_probability",
        )
        self.declare(
            OutputStatus(
                layer_index=layer_index,
                output_name="layer_result_printed",
                status="printed",
            ),
        )
