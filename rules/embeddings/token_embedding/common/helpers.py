def build_raw_token_shape(batch_size, sequence_length):
    return f"({batch_size}, {sequence_length})"


def build_token_embedding_matrix_shape(vocab_size, embedding_dim):
    return f"({vocab_size}, {embedding_dim})"


def build_token_embedding_output_shape(batch_size, sequence_length, embedding_dim):
    return f"({batch_size}, {sequence_length}, {embedding_dim})"


def build_token_embedding_parameter_shape(vocab_size, embedding_dim):
    return f"E: ({vocab_size}, {embedding_dim})"
