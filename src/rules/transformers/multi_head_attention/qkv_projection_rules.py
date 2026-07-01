from experta import Rule, MATCH, NOT, TEST

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    IntermediateTensor,
    CalculationStep
)


class MultiHeadAttentionQKVProjectionRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="multi_head_attention"
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
        TEST(lambda model_dim, num_heads: num_heads > 0),
        TEST(lambda model_dim, num_heads: model_dim % num_heads == 0),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            tensor_name="query_tensor"
        )),
        salience=64
    )
    def calculate_qkv_projection_shapes(
        self,
        layer_index,
        previous_layer_index,
        batch_size,
        sequence_length,
        model_dim,
        num_heads
    ):
        head_dim = model_dim // num_heads
        input_shape = f"({batch_size}, {sequence_length}, {model_dim})"
        qkv_shape = f"({batch_size}, {num_heads}, {sequence_length}, {head_dim})"

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="head_dim",
            value=head_dim
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="input_hidden_tensor",
            role="input",
            dims_count=3,
            shape=input_shape,
            data_type="float_hidden_tensor",
            source=f"layer_{previous_layer_index}_output_or_user_provided_hidden_tensor"
        ))

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="query_tensor",
            shape=qkv_shape,
            data_type="float_tensor",
            source="input_hidden_tensor_projected_by_WQ_and_split_into_heads"
        ))

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="key_tensor",
            shape=qkv_shape,
            data_type="float_tensor",
            source="input_hidden_tensor_projected_by_WK_and_split_into_heads"
        ))

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            tensor_name="value_tensor",
            shape=qkv_shape,
            data_type="float_tensor",
            source="input_hidden_tensor_projected_by_WV_and_split_into_heads"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_input_shape",
            previous_step_key="none",
            step_order=1,
            title="Multi-Head Attention input shape",
            formula="(batch_size, sequence_length, model_dim)",
            substitution=f"({batch_size}, {sequence_length}, {model_dim})",
            result=input_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_head_dim",
            previous_step_key=f"layer_{layer_index}_mha_input_shape",
            step_order=2,
            title="Attention head dimension",
            formula="head_dim = model_dim / num_heads",
            substitution=f"{model_dim} / {num_heads}",
            result=str(head_dim)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            step_key=f"layer_{layer_index}_mha_qkv_projection_shape",
            previous_step_key=f"layer_{layer_index}_mha_head_dim",
            step_order=3,
            title="Q, K, V projection shapes",
            formula="Q, K, V = (batch_size, num_heads, sequence_length, head_dim)",
            substitution=f"({batch_size}, {num_heads}, {sequence_length}, {head_dim})",
            result=qkv_shape
        ))
