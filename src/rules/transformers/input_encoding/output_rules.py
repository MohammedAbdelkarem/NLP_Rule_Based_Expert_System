from experta import Rule, MATCH, NOT

from src.facts import (
    Layer,
    ComponentGroup,
    IntermediateTensor,
    WeightShape,
    OutputStatus
)
from src.utils.print_helpers import print_grouped_layer_result


class InputEncodingOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="input_encoding",
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            status="computed"
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            weight_name="input_encoding_parameters",
            shape=MATCH.parameter_shape
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            tensor_name="token_embedding_output",
            shape=MATCH.token_embedding_output_shape
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            tensor_name="positional_embedding_output",
            shape=MATCH.positional_embedding_output_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_input_encoding_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        internal_layers,
        combine_operation,
        parameter_shape,
        token_embedding_output_shape,
        positional_embedding_output_shape
    ):
        print_grouped_layer_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="input_encoding",
            layer_kind="grouped_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            parameter_shape=parameter_shape,
            token_embedding_output_shape=token_embedding_output_shape,
            positional_embedding_output_shape=positional_embedding_output_shape
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
