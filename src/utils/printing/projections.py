from src.utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_weight_block,
    print_simple_stored_facts_block,
)


def print_position_wise_linear_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    output_data_type
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Position-wise Linear applies the same trainable projection to every token position.")
    print("Formula: Y = XW + b")

    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output tensor shape", output_shape, output_data_type)

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )
