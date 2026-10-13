from utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_weight_block,
    print_simple_stored_facts_block,
)


def print_token_embedding_result(
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
    print("Token Embedding maps integer token IDs to dense trainable vectors.")
    print("Operation: lookup each token ID as one row from the embedding matrix E.")

    print_shape_block("Input token ID shape", input_shape, "integer token indices")
    print_weight_block(parameter_shape)
    print_shape_block("Output embedding tensor shape", output_shape, output_data_type)

    print()
    print("Trainable status:")
    print("- has_weights = True")
    print("- trainable = True")
    print("- has_bias = False")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )


def print_pretrained_static_embedding_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    output_data_type,
    source_value,
    trainable_value
):
    print_layer_header(layer_index, layer_title)

    print()
    print("Pretrained Static Embedding maps token IDs to vectors loaded from a pretrained matrix.")
    print("Operation: lookup each token ID as one row from E_pretrained.")

    print_shape_block("Input token ID shape", input_shape, "integer token indices")
    print_weight_block(parameter_shape)
    print_shape_block("Output embedding tensor shape", output_shape, output_data_type)

    print()
    print("Pretrained source:")
    print(f"- source = {source_value}")

    print()
    print("Trainable status:")
    print("- has_weights = True")
    print(f"- trainable_mode = {trainable_value}")
    print("- has_bias = False")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )
