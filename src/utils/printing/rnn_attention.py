from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_rnn_attention_result(layer_index, layer_title, input_shape, output_shape, attention_weights_shape, parameter_shape, internal_layers, combine_operation):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a Chapter 13 encoder-decoder attention component.")
    print("It builds decoder context vectors by weighting encoder hidden states.")
    print("Internal components:")
    print(internal_layers)
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Input representation", input_shape, "encoder states plus decoder states")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block("Attention weights", attention_weights_shape, "alignment probabilities")
    print_shape_block("Context vector output", output_shape, "float context tensor")
    print_grouped_stored_facts_block(layer_index, "rnn_attention", "attention_component", input_shape, output_shape)
