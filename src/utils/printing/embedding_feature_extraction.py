from src.utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_count_based_embedding_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    vocab_size,
    context_vocab_size,
    context_window_size,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding feature-extraction component group.")
    print("It builds sparse word-context embedding vectors from corpus statistics.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Count-based embedding configuration:")
    print(f"Word vocabulary size: {vocab_size}")
    print(f"Context vocabulary size: {context_vocab_size}")
    print(f"Context window size: {context_window_size}")

    print_shape_block("Input representation", input_shape, "corpus distribution statistics")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Word-context vector matrix",
        shape=(
            "(vocab_size, context_vocab_size) = "
            f"({vocab_size}, {context_vocab_size})"
        ),
        data_type="sparse count embedding matrix"
    )
    print_shape_block("Output sparse embedding matrix", output_shape, "sparse count vectors")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="count_based_embedding",
        layer_kind="embedding_feature_extraction_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )


def print_co_occurrence_matrix_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    vocab_size,
    context_vocab_size,
    context_window_size,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is the Chapter 5 word-context co-occurrence matrix component group.")
    print("It is the core count matrix used by simple count-based embeddings.")
    print("It has no trainable weights and is not a neural network layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("Co-occurrence configuration:")
    print(f"Word vocabulary size: {vocab_size}")
    print(f"Context vocabulary size: {context_vocab_size}")
    print(f"Context window size: {context_window_size}")

    print_shape_block("Input representation", input_shape, "corpus distribution statistics")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Co-occurrence matrix",
        shape=(
            "(vocab_size, context_vocab_size) = "
            f"({vocab_size}, {context_vocab_size})"
        ),
        data_type="corpus count matrix"
    )
    print_shape_block("Output co-occurrence matrix", output_shape, "corpus count matrix")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="co_occurrence_matrix",
        layer_kind="corpus_statistics_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )
