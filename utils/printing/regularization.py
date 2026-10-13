from utils.printing.common import print_layer_header, print_shape_block, print_weight_block


def print_regularization_result(
    layer_index,
    layer_title,
    component_name,
    component_type,
    input_shape,
    output_shape,
    used_during,
    parameter_shape,
):
    print_layer_header(layer_index, layer_title)
    print()
    print("This is a training constraint, not a forward neural layer.")
    print(f"Component name: {component_name}")
    print(f"Component type: {component_type}")
    print(f"Used during: {used_during}")
    print()
    print_shape_block("Regularization input", input_shape, "trainable parameters")
    print_weight_block(parameter_shape)
    print_shape_block("Preserved forward output", output_shape, "same as previous model output")
    print()
    print("Regularization adds a penalty to the task loss:")
    print("total_loss = cross_entropy_loss + regularization_penalty")
    print()
    print("Parameter status:")
    print("- trainable = False")
    print("- has_weights = False")
    print("- updates no shape")
    print("- affects training objective only")
