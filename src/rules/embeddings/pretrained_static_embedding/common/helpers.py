from src.facts import (
    CalculationStep,
    TensorInfo,
    WeightShape,
)


def build_pretrained_embedding_parameter_shape(
    vocab_size,
    embedding_dim,
    source_value,
    trainable_value
):
    return (
        f"E_pretrained: ({vocab_size}, {embedding_dim}); "
        f"source={source_value}; "
        f"trainable={trainable_value}"
    )


def declare_pretrained_embedding_tensors(
    engine,
    layer_index,
    raw_token_shape,
    matrix_shape,
    output_shape,
    parameter_shape,
    source_value
):
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        weight_name="pretrained_embedding_matrix",
        shape=parameter_shape,
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="raw_token_ids",
        role="input",
        dims_count=2,
        shape=raw_token_shape,
        data_type="integer_token_indices",
        source="tokenizer_output",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="pretrained_embedding_matrix_shape",
        role="parameter_metadata",
        dims_count=2,
        shape=matrix_shape,
        data_type="float_embedding_matrix",
        source=source_value,
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="pretrained_static_embedding_output",
        role="output",
        dims_count=3,
        shape=output_shape,
        data_type="float_hidden_tensor",
        source="raw_token_ids_lookup_in_pretrained_embedding_matrix",
    ))


def declare_pretrained_embedding_shape_metadata(
    engine,
    layer_index,
    batch_size,
    sequence_length,
    embedding_dim,
    source_value,
    trainable_value
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source="batch_size_parameter",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(sequence_length),
        data_type="integer_dimension_value",
        source="sequence_length_parameter",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(embedding_dim),
        data_type="integer_dimension_value",
        source="embedding_dim_parameter",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="embedding_source",
        role="configuration_metadata",
        dims_count=1,
        shape=source_value,
        data_type="pretrained_embedding_source",
        source="user_selection",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        tensor_name="embedding_training_mode",
        role="configuration_metadata",
        dims_count=1,
        shape=trainable_value,
        data_type="embedding_trainability_mode",
        source="user_selection",
    ))


def declare_pretrained_embedding_calculation_steps(
    engine,
    layer_index,
    batch_size,
    sequence_length,
    vocab_size,
    embedding_dim,
    source_value,
    trainable_value,
    raw_token_shape,
    matrix_shape,
    output_shape
):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        step_key=f"layer_{layer_index}_raw_token_shape",
        previous_step_key="none",
        step_order=1,
        title="Raw token ID input shape",
        formula="(batch_size, sequence_length)",
        substitution=f"({batch_size}, {sequence_length})",
        result=raw_token_shape,
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        step_key=f"layer_{layer_index}_pretrained_matrix_shape",
        previous_step_key=f"layer_{layer_index}_raw_token_shape",
        step_order=2,
        title="Pretrained static embedding matrix shape",
        formula="(vocab_size, embedding_dim)",
        substitution=f"({vocab_size}, {embedding_dim}) from {source_value}",
        result=matrix_shape,
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        step_key=f"layer_{layer_index}_pretrained_trainability",
        previous_step_key=f"layer_{layer_index}_pretrained_matrix_shape",
        step_order=3,
        title="Pretrained embedding trainability mode",
        formula="frozen or fine_tunable",
        substitution=trainable_value,
        result=trainable_value,
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="pretrained_static_embedding",
        step_key=f"layer_{layer_index}_pretrained_output_shape",
        previous_step_key=f"layer_{layer_index}_pretrained_trainability",
        step_order=4,
        title="Pretrained static embedding output shape",
        formula="(batch_size, sequence_length, embedding_dim)",
        substitution=f"({batch_size}, {sequence_length}, {embedding_dim})",
        result=output_shape,
    ))
