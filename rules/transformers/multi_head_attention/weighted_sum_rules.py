from experta import Rule, MATCH, NOT

from facts import LayerParameter, IntermediateTensor, CalculationStep


class MultiHeadAttentionWeightedSumRules:
    @Rule(
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="batch_size",
            value=MATCH.batch_size
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
            parameter_name="sequence_length",
            value=MATCH.sequence_length
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
            tensor_name="attention_weights"
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="value_tensor"
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="head_output"
        )),
        salience=61
    )
    def calculate_weighted_sum_shape(
        self,
        layer_index,
        batch_size,
        num_heads,
        sequence_length,
        head_dim
    ):
        head_output_shape = f"({batch_size}, {num_heads}, {sequence_length}, {head_dim})"

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="head_output",
            shape=head_output_shape,
            data_type="float_tensor",
            source="attention_weights_times_value_tensor"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_head_output_shape",
            previous_step_key=f"layer_{layer_index}_mha_attention_weights_shape",
            step_order=6,
            title="Weighted sum output shape for all heads",
            formula="attention_weights × V = (batch_size, num_heads, sequence_length, head_dim)",
            substitution=f"({batch_size}, {num_heads}, {sequence_length}, {head_dim})",
            result=head_output_shape
        ))
