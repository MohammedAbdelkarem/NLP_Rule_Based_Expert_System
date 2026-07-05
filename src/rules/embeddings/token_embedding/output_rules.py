from experta import Rule, MATCH, NOT

from src.facts import Layer, WeightShape, TensorInfo, OutputStatus
from src.utils.print_helpers import print_token_embedding_result


class TokenEmbeddingOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
            weight_name="token_embedding_matrix",
            shape=MATCH.parameter_shape
        ),
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding",
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
    def print_token_embedding_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        parameter_shape,
        output_data_type
    ):
        print_token_embedding_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="token_embedding",
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
