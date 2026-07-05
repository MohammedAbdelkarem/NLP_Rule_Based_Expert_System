from src.facts import (
    CalculationStep,
    ComponentGroup,
    LayerMetadata,
    TensorInfo,
    WeightShape,
)


def matrix_shape(row_count, column_count):
    return f"({int(row_count)}, {int(column_count)})"


def vector_bank_shape(item_count, vector_dim):
    return f"({int(item_count)}, {int(vector_dim)})"


def declare_feature_component_group(
    engine,
    layer_index,
    group_name,
    group_title,
    layer_kind,
    layer_role,
    internal_layers,
    combine_operation,
    input_shape,
    output_shape,
    tensor_name,
    tensor_type,
    parameter_name,
    parameter_shape,
):
    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name=group_name,
        group_title=group_title,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
        input_shape=input_shape,
        output_shape=output_shape,
        status="computed"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name=tensor_name,
        role="output",
        dims_count=2,
        shape=output_shape,
        data_type=tensor_type,
        source=group_name
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=group_name,
        weight_name=parameter_name,
        shape=parameter_shape
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=group_name,
        layer_kind=layer_kind,
        layer_role=layer_role,
        repeat_policy="allowed_with_new_corpus_or_new_window",
        consecutive_repeat_policy="not_recommended",
        terminal_status="non_terminal"
    ))


def declare_feature_shape_metadata(
    engine,
    layer_index,
    layer_name,
    vocab_size,
    context_vocab_size,
    context_window_size,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="vocab_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(vocab_size)),
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
        tensor_name="context_window_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(context_window_size)),
        data_type="integer_dimension_value",
        source=layer_name
    ))


def declare_count_based_steps(
    engine,
    layer_index,
    vocab_size,
    context_vocab_size,
    output_shape,
):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="count_based_embedding",
        step_key="count_based_matrix_shape",
        previous_step_key="input_corpus_statistics",
        step_order=1,
        title="Build sparse word-context vectors",
        formula="word_context_matrix_shape = (vocab_size, context_vocab_size)",
        substitution=(
            f"word_context_matrix_shape = "
            f"({int(vocab_size)}, {int(context_vocab_size)})"
        ),
        result=f"word_context_matrix_shape = {output_shape}"
    ))


def declare_cooccurrence_steps(
    engine,
    layer_index,
    vocab_size,
    context_vocab_size,
    output_shape,
):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="co_occurrence_matrix",
        step_key="co_occurrence_matrix_shape",
        previous_step_key="context_window_statistics",
        step_order=1,
        title="Build word-context co-occurrence matrix",
        formula="co_occurrence_matrix_shape = (vocab_size, context_vocab_size)",
        substitution=(
            f"co_occurrence_matrix_shape = "
            f"({int(vocab_size)}, {int(context_vocab_size)})"
        ),
        result=f"co_occurrence_matrix_shape = {output_shape}"
    ))
