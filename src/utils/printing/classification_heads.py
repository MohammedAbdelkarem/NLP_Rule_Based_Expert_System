from src.utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_simple_stored_facts_block,
    print_weight_block,
)


def print_logistic_regression_head_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    internal_layers,
    combine_operation,
    input_shape,
    output_shape,
    parameter_shape,
    output_data_type,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a Chapter 4 classifier head, not a hidden feature-extraction block.")
    print(f"Internal components: {internal_layers}")
    print(f"Combine operation: {combine_operation}")
    print("Role: trainable output head for supervised classification.")
    print()
    print_shape_block("Input feature shape", input_shape, "float_hidden_tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output probability shape", output_shape, output_data_type)
    print()
    print("Parameter status:")
    print("- trainable = True")
    print("- has_weights = True")
    print("- has_bias = True")
    print("- used_during_inference = True")
    print("- loss is not included here; add the correct Cross-Entropy loss during training")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape,
    )
