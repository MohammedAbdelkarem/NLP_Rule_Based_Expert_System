from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_embedding_visualization_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    num_items,
    embedding_dim,
    visualization_dim,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding analysis component.")
    print("It prepares low-dimensional coordinates to inspect embedding neighborhoods.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Visualization configuration:")
    print(f"Embedded items: {num_items}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Visualization dimension: {visualization_dim}")

    print_shape_block("Input embedding matrix", input_shape, "dense embedding matrix")
    print_weight_block(parameter_shape)
    print_shape_block("Output coordinate matrix", output_shape, "visualization coordinates")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="embedding_visualization",
        layer_kind="embedding_analysis_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_embedding_bias_analysis_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    target_word_count,
    attribute_word_count,
    embedding_dim,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding bias analysis component.")
    print("It compares target word vectors with attribute word vectors.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Bias analysis configuration:")
    print(f"Target words: {target_word_count}")
    print(f"Attribute words: {attribute_word_count}")
    print(f"Embedding dimension: {embedding_dim}")

    print_shape_block("Input embedding subsets", input_shape, "dense embedding vectors")
    print_weight_block(parameter_shape)
    print_shape_block("Output association matrix", output_shape, "bias score matrix")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="embedding_bias_analysis",
        layer_kind="embedding_bias_analysis_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_embedding_evaluation_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    evaluation_pair_count,
    embedding_dim,
    benchmark_count,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding evaluation component.")
    print("It scores embedding quality with similarity or analogy benchmark tasks.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Evaluation configuration:")
    print(f"Evaluation pairs: {evaluation_pair_count}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Benchmarks: {benchmark_count}")

    print_shape_block("Input evaluation vectors", input_shape, "dense embedding vectors")
    print_weight_block(parameter_shape)
    print_shape_block("Output evaluation report", output_shape, "evaluation score matrix")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="embedding_evaluation",
        layer_kind="embedding_evaluation_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )
