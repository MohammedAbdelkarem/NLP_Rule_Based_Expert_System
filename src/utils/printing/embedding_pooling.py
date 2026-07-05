from src.utils.printing.common import (
    print_layer_header,
    print_shape_block,
    print_simple_stored_facts_block,
    print_weight_block,
)


def print_embedding_pooling_adapter_result(
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
    print("This adapter pools token embeddings into one vector per example.")
    print("Operation: mean over the sequence axis")
    print_shape_block("Input embedding sequence shape", input_shape, "float hidden tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output classifier feature shape", output_shape, "float hidden tensor")
    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape,
    )
