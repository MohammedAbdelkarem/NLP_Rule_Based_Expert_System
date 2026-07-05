from src.utils.printing.common import print_layer_header, print_shape_block, print_weight_block


def print_loss_component_result(
    layer_index,
    layer_title,
    component_name,
    component_type,
    input_shape,
    output_shape,
    used_during,
    parameter_shape,
    loss_family,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a training component, not an inference layer.")
    print(f"Component name: {component_name}")
    print(f"Component type: {component_type}")
    print(f"Loss family: {loss_family}")
    print(f"Used during: {used_during}")
    print()
    print_shape_block("Loss inputs", input_shape, "predictions + labels")
    print_weight_block(parameter_shape)
    print_shape_block("Loss output", output_shape, "scalar_loss")
    print()
    print("Parameter status:")
    print("- trainable = False")
    print("- has_weights = False")
    print("- used_during_training = True")
    print("- used_during_inference = False")
