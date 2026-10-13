from facts import CalculationStep, ComponentGroup, LayerMetadata, TensorInfo, WeightShape


def embedding_matrix_shape(vocab_size, embedding_dim):
    return f"({int(vocab_size)}, {int(embedding_dim)})"


def declare_training_shape_metadata(
    engine,
    layer_index,
    layer_name,
    target_vocab_size,
    context_vocab_size,
    embedding_dim,
    context_window_size,
    negative_sample_count,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="target_vocab_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(target_vocab_size)),
        data_type="integer_dimension_value",
        source=layer_name
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="context_vocab_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(context_vocab_size)),
        data_type="integer_dimension_value",
        source=layer_name
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="embedding_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(embedding_dim)),
        data_type="integer_dimension_value",
        source=layer_name
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="context_window_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(context_window_size)),
        data_type="integer_dimension_value",
        source=layer_name
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="negative_sample_count",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(negative_sample_count)),
        data_type="integer_dimension_value",
        source=layer_name
    ))


def declare_sgns_component(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    parameter_shape,
):
    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name="skipgram_sgns",
        group_title=layer_title,
        internal_layers=(
            "target embedding lookup + context embedding lookup + dot product score + "
            "sigmoid binary decision + negative sampling loss"
        ),
        combine_operation=(
            "positive target-context pairs and sampled negative pairs update "
            "the embedding matrices"
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        status="computed"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        tensor_name="learned_static_embedding_matrix",
        role="output",
        dims_count=2,
        shape=output_shape,
        data_type="learned_static_embedding_matrix",
        source="skipgram_sgns"
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        weight_name="trainable_embedding_matrices",
        shape=parameter_shape
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        layer_kind="embedding_training_component_group",
        layer_role="learn_static_embeddings_from_context_prediction",
        repeat_policy="allowed_with_new_corpus_or_new_hyperparameters",
        consecutive_repeat_policy="not_recommended",
        terminal_status="terminal"
    ))


def declare_sgns_steps(
    engine,
    layer_index,
    target_vocab_size,
    context_vocab_size,
    embedding_dim,
    output_shape,
):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        step_key="sgns_embedding_matrix_shape",
        previous_step_key="skipgram_training_pairs",
        step_order=1,
        title="Learn static embedding matrix using SGNS",
        formula="embedding_matrix_shape = (target_vocab_size, embedding_dim)",
        substitution=(
            f"embedding_matrix_shape = "
            f"({int(target_vocab_size)}, {int(embedding_dim)})"
        ),
        result=f"learned_static_embedding_matrix = {output_shape}"
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="skipgram_sgns",
        step_key="sgns_context_matrix_shape",
        previous_step_key="sgns_embedding_matrix_shape",
        step_order=2,
        title="Declare context embedding matrix",
        formula="context_matrix_shape = (context_vocab_size, embedding_dim)",
        substitution=(
            f"context_matrix_shape = "
            f"({int(context_vocab_size)}, {int(embedding_dim)})"
        ),
        result="context matrix participates during training only"
    ))
