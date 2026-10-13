from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    IntermediateTensor,
    WeightShape,
    Layer,
    ComponentGroup,
    LayerMetadata,
    CalculationStep,
    TransformerStage
)


class InputEncodingSummationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="input_encoding",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="vocab_size",
            value=MATCH.vocab_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="model_dim",
            value=MATCH.model_dim
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
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def combine_embedding_outputs(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        vocab_size,
        model_dim,
        token_embedding_output_shape,
        positional_embedding_output_shape
    ):
        raw_token_shape = f"({batch_size}, {sequence_length})"
        token_embedding_matrix_shape = f"({vocab_size}, {model_dim})"
        positional_embedding_matrix_shape = f"({sequence_length}, {model_dim})"
        combined_output_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        parameter_shape = (
            f"token_embedding_matrix: {token_embedding_matrix_shape}, "
            f"positional_embedding_matrix: {positional_embedding_matrix_shape}"
        )
        combine_operation = (
            "Element-wise summation of token embedding output and "
            "positional embedding output to produce Transformer input X"
        )

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="input_encoding",
            weight_name="input_encoding_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="combined_transformer_input_x",
            role="output",
            dims_count=3,
            shape=combined_output_shape,
            data_type="float_hidden_tensor",
            source="element_wise_sum_token_embedding_output_and_positional_embedding_output"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="batch_size_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="sequence_length_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="input_encoding",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source="model_dim_parameter"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="input_encoding",
            group_title="Input Encoding Layer Group",
            internal_layers="Token Embedding + Positional Embedding",
            combine_operation=combine_operation,
            input_shape=raw_token_shape,
            output_shape=combined_output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="input_encoding",
            layer_kind="grouped_layer",
            layer_role="input_preparation",
            repeat_policy="once_first_only",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="input_encoding",
            stage_name="input_encoding_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="input_encoding",
            step_key=f"layer_{layer_index}_combined_transformer_input_shape",
            previous_step_key=f"layer_{layer_index}_positional_embedding_output_shape",
            step_order=6,
            title="Combined Transformer input X shape",
            formula="token_embedding_output + positional_embedding_output keeps the same shape",
            substitution=f"{token_embedding_output_shape} + {positional_embedding_output_shape}",
            result=combined_output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="input_encoding",
            layer_title=layer_title,
            input_shape=raw_token_shape,
            output_shape=combined_output_shape
        ))
