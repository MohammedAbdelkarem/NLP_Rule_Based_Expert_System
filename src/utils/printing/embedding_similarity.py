from src.utils.printing.common import (
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_simple_stored_facts_block,
    print_weight_block,
)


def print_cosine_similarity_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    vector_dim,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding similarity component.")
    print("It compares two vectors using normalized dot product.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Cosine similarity configuration:")
    print(f"Vector dimension: {vector_dim}")

    print_shape_block("Input vector pair", input_shape, "embedding vectors")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Similarity score shape",
        shape="(1,)",
        data_type="single float similarity score"
    )
    print_shape_block("Output similarity score", output_shape, "float similarity score")

    print_simple_stored_facts_block(
        layer_index=layer_index,
        layer_name="cosine_similarity",
        layer_kind="embedding_similarity_component_not_neural_layer",
        input_shape=input_shape,
        output_shape=output_shape,
        parameter_shape=parameter_shape
    )
