from src.utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_weight_block,
    print_simple_stored_facts_block,
)


def print_activation_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Activation layers apply an element-wise non-linear transformation.")
    print(f"Activation name: {layer_name.upper()}")

    print_shape_block("Input tensor shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output tensor shape", output_shape, "float hidden tensor")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_softmax_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Softmax converts logits into probabilities over the last dimension.")
    print("Input data type: logits")

    print_shape_block("Input logits shape", input_shape, "logits")
    print_weight_block(parameter_shape)
    print_shape_block("Output probability shape", output_shape, "probabilities")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_sigmoid_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Sigmoid converts one binary logit into a probability between 0 and 1.")
    print("It is used by binary logistic regression heads.")

    print_shape_block("Input binary logits shape", input_shape, "binary_logits")
    print_weight_block(parameter_shape)
    print_shape_block("Output binary probability shape", output_shape, "binary_probability")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape,
    )
