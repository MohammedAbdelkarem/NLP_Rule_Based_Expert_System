from experta import MATCH, NOT, Rule

from facts import OutputStatus, TrainingComponent, WeightShape
from utils.print_helpers import print_loss_component_result


class BinaryCrossEntropyOutputRules:
    @Rule(
        TrainingComponent(
            component_index=MATCH.layer_index,
            component_name="binary_cross_entropy",
            component_title=MATCH.layer_title,
            component_type=MATCH.component_type,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed",
            used_during=MATCH.used_during,
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            weight_name="loss_parameters",
            shape=MATCH.parameter_shape,
        ),
        NOT(
            OutputStatus(
                layer_index=MATCH.layer_index,
                output_name="layer_result_printed",
                status="printed",
            ),
        ),
        salience=50
    )
    def print_bce_result(
        self,
        layer_index,
        layer_title,
        component_type,
        input_shape,
        output_shape,
        used_during,
        parameter_shape
    ):
        print_loss_component_result(
            layer_index,
            layer_title,
            "binary_cross_entropy",
            component_type,
            input_shape,
            output_shape,
            used_during,
            parameter_shape,
            "binary_cross_entropy",
        )
        self.declare(
            OutputStatus(
                layer_index=layer_index,
                output_name="layer_result_printed",
                status="printed",
            ),
        )
