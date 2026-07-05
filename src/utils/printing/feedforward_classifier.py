from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_feedforward_classifier_head_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
    classifier_mode,
    hidden_dim,
    num_classes,
    parameter_shape,
    hidden_projection_shape,
    activation_output_shape,
    logits_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This layer is a terminal Chapter 6 feedforward classifier head.")
    print("Internal operations:")
    print(internal_layers)
    print_shape_block("Input feature vector shape", input_shape, "float hidden tensor")
    print()
    print("Classifier configuration:")
    print(f"Classifier mode: {classifier_mode}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of output classes: {num_classes}")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Hidden projection output shape",
        shape=hidden_projection_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Activation output shape",
        shape=activation_output_shape,
        data_type="float hidden tensor",
    )
    print_intermediate_tensor_block(
        title="Logits output shape",
        shape=logits_shape,
        data_type="logits",
    )
    print()
    print("Combine operation:")
    print(combine_operation)
    print_shape_block("Final classifier probability shape", output_shape, "probabilities")
    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
    )
