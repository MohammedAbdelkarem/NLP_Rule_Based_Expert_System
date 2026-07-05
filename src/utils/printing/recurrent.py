from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_simple_rnn_result(layer_index, layer_title, input_shape, output_shape, parameter_shape):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a real Chapter 13 recurrent neural network layer.")
    print("It carries a hidden state from one time step to the next.")
    print("It has trainable input weights, recurrent hidden weights, and bias.")
    print_shape_block("Input sequence tensor", input_shape, "float sequence tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output recurrent tensor", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(layer_index, "simple_rnn", "recurrent_layer", input_shape, output_shape)


def print_lstm_result(layer_index, layer_title, input_shape, output_shape, parameter_shape):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a real Chapter 13 gated recurrent neural network layer.")
    print("It uses a hidden state, a cell state, and forget/input/output gates.")
    print("The four gate computations are stored together in combined matrices.")
    print_shape_block("Input sequence tensor", input_shape, "float sequence tensor")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block("Cell state", "(batch_size, hidden_size)", "float cell state")
    print_shape_block("Output recurrent tensor", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(layer_index, "lstm", "recurrent_layer", input_shape, output_shape)


def print_stacked_rnn_result(layer_index, layer_title, input_shape, output_shape, parameter_shape, internal_layers, combine_operation):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a real Chapter 13 recurrent architecture component group.")
    print("It stacks recurrent layers so higher layers receive lower-layer sequences.")
    print("Internal components:")
    print(internal_layers)
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Input sequence tensor", input_shape, "float sequence tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output recurrent tensor", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(layer_index, "stacked_rnn", "recurrent_component_group", input_shape, output_shape)


def print_bidirectional_rnn_result(layer_index, layer_title, input_shape, output_shape, parameter_shape, internal_layers, combine_operation):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a real Chapter 13 bidirectional recurrent architecture group.")
    print("It uses one recurrent path for past context and one for future context.")
    print("Internal components:")
    print(internal_layers)
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Input sequence tensor", input_shape, "float sequence tensor")
    print_weight_block(parameter_shape)
    print_shape_block("Output merged recurrent tensor", output_shape, "float hidden tensor")
    print_grouped_stored_facts_block(layer_index, "bidirectional_rnn", "recurrent_component_group", input_shape, output_shape)
