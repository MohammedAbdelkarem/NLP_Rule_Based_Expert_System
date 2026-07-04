def print_section_title(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_label_value(label, value):
    print(f"{label}:")
    print(value)


def print_layer_header(layer_index, layer_title):
    print_section_title(f"Layer {layer_index}: {layer_title}")


def print_shape_block(title, shape, data_type):
    print()
    print_label_value(title, shape)
    print(f"Data type: {data_type}")


def print_shape_block_without_type(title, shape):
    print()
    print_label_value(title, shape)


def print_weight_block(parameter_shape):
    print()
    print("Learnable parameter shapes:")
    print(parameter_shape)


def print_simple_stored_facts_block(
    layer_index,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape
):
    print()
    print("Stored Facts:")
    print(f"Layer index: {layer_index}")
    print(f"Layer name: {layer_name}")
    print(f"Layer kind: {layer_kind}")
    print(f"Input shape: {input_shape}")
    print(f"Learnable parameters: {parameter_shape}")
    print(f"Output shape: {output_shape}")
    print()


def print_grouped_stored_facts_block(
    layer_index,
    layer_name,
    layer_kind,
    input_shape,
    output_shape
):
    print()
    print("Stored Facts:")
    print(f"Layer index: {layer_index}")
    print(f"Layer name: {layer_name}")
    print(f"Layer kind: {layer_kind}")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    print()


def print_intermediate_tensor_block(title, shape, data_type):
    print()
    print(f"{title}:")
    print(shape)
    print(f"Data type: {data_type}")


def print_simple_layer_result(
    layer_index,
    layer_title,
    layer_name,
    layer_kind,
    input_shape,
    output_shape,
    parameter_shape,
    data_type
):
    print_layer_header(layer_index, layer_title)
    print_shape_block("Input shape", input_shape, data_type)
    print_weight_block(parameter_shape)
    print_shape_block("Output shape", output_shape, data_type)
    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind=layer_kind,
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )
