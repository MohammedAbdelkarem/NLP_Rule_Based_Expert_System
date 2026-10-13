from utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_shape_block_without_type,
    print_weight_block,
    print_intermediate_tensor_block,
    print_simple_stored_facts_block,
    print_grouped_stored_facts_block,
)


def print_grouped_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    parameter_shape,
    token_embedding_output_shape,
    positional_embedding_output_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped layer.")
    print("Internal layers:")
    print(internal_layers)

    print_shape_block_without_type("Input data type", "integer token IDs")
    print_shape_block_without_type("Raw token IDs shape", input_shape)

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block(
        title="Token Embedding output shape",
        shape=token_embedding_output_shape,
        data_type="float tensor"
    )
    print_intermediate_tensor_block(
        title="Positional Embedding output shape",
        shape=positional_embedding_output_shape,
        data_type="float tensor"
    )

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Transformer input X shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_multi_head_attention_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    num_heads,
    head_dim,
    parameter_shape,
    query_shape,
    key_shape,
    value_shape,
    attention_scores_shape,
    attention_weights_shape,
    head_output_shape,
    concatenated_heads_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped attention layer.")
    print("Internal operations:")
    print(internal_layers)

    print_shape_block("Input hidden tensor shape", input_shape, "float hidden tensor")

    print()
    print("Attention configuration:")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block("Query tensor Q shape", query_shape, "float tensor")
    print_intermediate_tensor_block("Key tensor K shape", key_shape, "float tensor")
    print_intermediate_tensor_block("Value tensor V shape", value_shape, "float tensor")
    print_intermediate_tensor_block("Attention scores Q × K.T shape", attention_scores_shape, "float tensor")
    print_intermediate_tensor_block("Attention weights softmax output shape", attention_weights_shape, "float tensor")
    print_intermediate_tensor_block("Weighted sum head output shape", head_output_shape, "float tensor")
    print_intermediate_tensor_block("Concatenated heads shape", concatenated_heads_shape, "float tensor")

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Multi-Head Attention output shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_residual_connection_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    source_layer_index,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Residual Connection adds two compatible tensors element-wise.")
    print(f"Residual source layer index: {source_layer_index}")

    print_shape_block("Input tensors", input_shape, "float hidden tensor")

    print_weight_block(parameter_shape)

    print()
    print("Operation:")
    print("previous_layer_output + residual_source_tensor")

    print_shape_block("Residual output shape", output_shape, "float hidden tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_feed_forward_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    feed_forward_dim,
    parameter_shape,
    first_projection_shape,
    activation_output_shape,
    second_projection_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This layer is a grouped feed-forward layer.")
    print("Internal operations:")
    print(internal_layers)

    print_shape_block("Input hidden tensor shape", input_shape, "float hidden tensor")

    print()
    print("Feed-Forward configuration:")
    print(f"Feed-forward hidden dimension d_ff: {feed_forward_dim}")
    print(f"Selected activation: {activation_name.upper()}")

    print_weight_block(parameter_shape)

    print_intermediate_tensor_block("First projection output shape", first_projection_shape, "float hidden tensor")
    print_intermediate_tensor_block("Activation output shape", activation_output_shape, "float hidden tensor")
    print_intermediate_tensor_block("Second projection output shape", second_projection_shape, "float hidden tensor")

    print()
    print("Combine operation:")
    print(combine_operation)

    print_shape_block("Final Feed-Forward output shape", output_shape, "float hidden tensor")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape
    )
