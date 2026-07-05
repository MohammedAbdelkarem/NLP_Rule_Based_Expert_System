from experta import MATCH, NOT, Rule

from src.facts import Layer, OutputStatus, WeightShape
from src.utils.print_helpers import print_embedding_pooling_adapter_result


class EmbeddingToClassifierPoolingOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            weight_name="pooling_parameters",
            shape=MATCH.parameter_shape,
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed",
        )),
        salience=50,
    )
    def print_pooling_adapter_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape,
    ):
        print_embedding_pooling_adapter_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="embedding_to_classifier_pooling",
            layer_kind="adapter_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
        )
        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed",
        ))
