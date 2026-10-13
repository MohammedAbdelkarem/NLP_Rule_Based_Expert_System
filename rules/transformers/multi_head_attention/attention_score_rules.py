from experta import Rule, MATCH, NOT

from facts import LayerParameter, IntermediateTensor, CalculationStep


class MultiHeadAttentionScoreRules:
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
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="query_tensor"
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="key_tensor"
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="attention_scores"
        )),
        salience=63
    )
    def calculate_attention_score_shape(
        self,
        layer_index,
        batch_size,
        num_heads,
        sequence_length
    ):
        attention_score_shape = f"({batch_size}, {num_heads}, {sequence_length}, {sequence_length})"

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="attention_scores",
            shape=attention_score_shape,
            data_type="float_tensor",
            source="query_tensor_times_key_tensor_transposed_scaled_by_sqrt_head_dim"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_attention_scores_shape",
            previous_step_key=f"layer_{layer_index}_mha_qkv_projection_shape",
            step_order=4,
            title="Attention score tensor shape",
            formula="Q × K^T = (batch_size, num_heads, sequence_length, sequence_length)",
            substitution=f"({batch_size}, {num_heads}, {sequence_length}, {sequence_length})",
            result=attention_score_shape
        ))
