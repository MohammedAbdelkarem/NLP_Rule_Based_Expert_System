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


class MultiHeadAttentionOutputProjectionRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            layer_title=MATCH.layer_title
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=MATCH.num_heads
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="head_dim",
            value=MATCH.head_dim
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="head_output",
            shape=MATCH.head_output_shape
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_output_projection_shape(
        self,
        layer_index,
        layer_title,
        batch_size,
        sequence_length,
        model_dim,
        num_heads,
        head_dim,
        head_output_shape
    ):
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        concatenated_heads_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        concatenated_heads_substitution = f"({batch_size}, {sequence_length}, {num_heads} × {head_dim}) = {concatenated_heads_shape}"
        output_shape = input_shape
        parameter_shape = (
            f"WQ: ({model_dim}, {model_dim}), "
            f"WK: ({model_dim}, {model_dim}), "
            f"WV: ({model_dim}, {model_dim}), "
            f"WO: ({model_dim}, {model_dim})"
        )
        combine_operation = (
            "Parallel attention heads are concatenated, then the output projection WO "
            "maps the result back to model_dim."
        )

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="concatenated_heads",
            shape=concatenated_heads_shape,
            data_type="float_tensor",
            source="concatenate_all_attention_head_outputs"
        ))

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            weight_name="multi_head_attention_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="output_attention_tensor",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_hidden_tensor",
            source="concatenated_heads_projected_by_WO"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="multi_head_attention_input_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(sequence_length),
            data_type="integer_dimension_value",
            source="multi_head_attention_input_sequence_length"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=str(model_dim),
            data_type="integer_dimension_value",
            source="multi_head_attention_output_projection_dimension"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="multi_head_attention",
            group_title="Multi-Head Attention Layer Group",
            internal_layers=(
                "QKV Projection + Scaled Dot-Product Attention + Attention Softmax + "
                "Weighted Sum + Head Concatenation + Output Projection"
            ),
            combine_operation=combine_operation,
            input_shape=input_shape,
            output_shape=output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            layer_kind="grouped_layer",
            layer_role="attention",
            repeat_policy="repeat_allowed",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(TransformerStage(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            stage_name="attention_completed",
            status="computed"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_concatenated_heads_shape",
            previous_step_key=f"layer_{layer_index}_mha_head_output_shape",
            step_order=7,
            title="Concatenated heads shape",
            formula="(batch_size, sequence_length, num_heads × head_dim)",
            substitution=concatenated_heads_substitution,
            result=concatenated_heads_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_output_projection_shape",
            previous_step_key=f"layer_{layer_index}_mha_concatenated_heads_shape",
            step_order=8,
            title="Output projection shape",
            formula="WO maps concatenated heads back to (batch_size, sequence_length, model_dim)",
            substitution=f"WO: ({model_dim}, {model_dim})",
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
