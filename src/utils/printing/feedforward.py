from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_hidden_dense_activation_block_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    hidden_dim,
    parameter_shape,
    projection_shape,
    activation_output_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a grouped Chapter 6 hidden dense block.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print()
    print("Hidden block configuration:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Selected activation: {activation_name.upper()}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Dense projection output shape",
        shape=projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final hidden block output shape", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def print_feedforward_mlp_block_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    activation_name,
    hidden_dim,
    parameter_shape,
    first_projection_shape,
    activation_output_shape,
    second_projection_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a generic Chapter 6 feedforward MLP block.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print()
    print("MLP configuration:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Selected activation: {activation_name.upper()}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="First projection output shape",
        shape=first_projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Second projection output shape",
        shape=second_projection_shape,
        data_type="float hidden tensor",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final MLP output shape", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )
