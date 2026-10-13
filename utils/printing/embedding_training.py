from utils.printing.common import (
    print_grouped_stored_facts_block,
    print_intermediate_tensor_block,
    print_layer_header,
    print_shape_block,
    print_weight_block,
)


def print_skipgram_sgns_result(
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
    internal_layers,
    combine_operation,
    target_vocab_size,
    context_vocab_size,
    embedding_dim,
    context_window_size,
    negative_sample_count,
):
    print_layer_header(layer_index, layer_title)

    print()
    print("This is a Chapter 5 embedding training component group.")
    print("It trains static word embeddings with skip-gram and negative sampling.")
    print("It is not a normal inference layer.")

    print()
    print("Internal components:")
    print(internal_layers)

    print()
    print("Combine operation:")
    print(combine_operation)

    print()
    print("SGNS configuration:")
    print(f"Target vocabulary size: {target_vocab_size}")
    print(f"Context vocabulary size: {context_vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Context window size: {context_window_size}")
    print(f"Negative samples: {negative_sample_count}")

    print_shape_block("Input training pairs", input_shape, "target-context pair set")
    print_weight_block(parameter_shape)
    print_intermediate_tensor_block(
        title="Target embedding matrix",
        shape=f"({target_vocab_size}, {embedding_dim})",
        data_type="trainable static embedding matrix"
    )
    print_intermediate_tensor_block(
        title="Context embedding matrix",
        shape=f"({context_vocab_size}, {embedding_dim})",
        data_type="trainable context embedding matrix"
    )
    print_shape_block("Output learned static embedding matrix", output_shape, "embedding matrix")

    print_grouped_stored_facts_block(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        layer_kind="embedding_training_component_group",
        input_shape=input_shape,
        output_shape=output_shape
    )
