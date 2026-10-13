from experta import Rule, MATCH, NOT

from facts import IntermediateTensor, CalculationStep


class MultiHeadAttentionSoftmaxRules:
    @Rule(
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="attention_scores",
            shape=MATCH.attention_score_shape
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="attention_weights"
        )),
        salience=62
    )
    def calculate_attention_weight_shape(
        self,
        layer_index,
        attention_score_shape
    ):
        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="attention_weights",
            shape=attention_score_shape,
            data_type="float_tensor",
            source="softmax_applied_to_scaled_attention_scores"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_attention_weights_shape",
            previous_step_key=f"layer_{layer_index}_mha_attention_scores_shape",
            step_order=5,
            title="Attention weights shape",
            formula="softmax(attention_scores) keeps the same shape",
            substitution=attention_score_shape,
            result=attention_score_shape
        ))
