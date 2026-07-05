from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_rnn_encoder_decoder_result(layer_index, layer_title, input_shape, output_shape, parameter_shape, internal_layers, combine_operation):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a Chapter 13 sequence-to-sequence recurrent component group.")
    print("The encoder represents the source sequence, and the decoder generates target states.")
    print("Internal components:")
    print(internal_layers)
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Encoder input sequence", input_shape, "float source sequence tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Decoder output sequence", output_shape, "float decoder sequence tensor")
    print_grouped_stored_facts_block(layer_index, "rnn_encoder_decoder", "sequence_to_sequence_component_group", input_shape, output_shape)
