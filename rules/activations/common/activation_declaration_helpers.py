from facts import (
    WeightShape,
    TensorInfo,
    LayerMetadata,
    TransformerStage,
    CalculationStep,
    Layer
)


def declare_hidden_activation_layer(
    engine,
    layer_index,
    layer_name,
    layer_title,
    dims_count,
    input_shape,
    batch_size,
    sequence_length,
    model_dim,
    batch_source,
    sequence_source,
    model_dim_source,
    formula,
    substitution
):
    output_shape = input_shape
    parameter_shape = "no trainable parameters"

    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="activation_parameters",
        shape=parameter_shape
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name=f"{layer_name}_output",
        role="output",
        dims_count=dims_count,
        shape=output_shape,
        data_type="float_hidden_tensor",
        source=f"{layer_name}_activation"
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source=batch_source
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(sequence_length),
        data_type="integer_dimension_value",
        source=sequence_source
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(model_dim),
        data_type="integer_dimension_value",
        source=model_dim_source
    ))

    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="simple_layer",
        layer_role="activation",
        repeat_policy="repeat_allowed",
        consecutive_repeat_policy="no_consecutive_activation",
        terminal_status="non_terminal"
    ))

    engine.declare(TransformerStage(
        layer_index=layer_index,
        layer_name=layer_name,
        stage_name=f"{layer_name}_activation_completed",
        status="computed"
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_activation_shape",
        previous_step_key="none",
        step_order=1,
        title=f"{layer_title} output shape",
        formula=formula,
        substitution=substitution,
        result=output_shape
    ))

    engine.declare(Layer(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape
    ))


def declare_probability_activation_layer(
    engine,
    layer_index,
    layer_name,
    layer_title,
    input_shape,
    batch_size,
    sequence_length,
    vector_dim,
    batch_source,
    sequence_source,
    vector_dim_source,
    substitution
):
    output_shape = input_shape
    parameter_shape = "no trainable parameters"

    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="softmax_parameters",
        shape=parameter_shape
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="softmax_output",
        role="output",
        dims_count=3,
        shape=output_shape,
        data_type="probabilities",
        source="softmax_probability_activation"
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source=batch_source
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(sequence_length),
        data_type="integer_dimension_value",
        source=sequence_source
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(vector_dim),
        data_type="integer_dimension_value",
        source=vector_dim_source
    ))

    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="simple_layer",
        layer_role="activation",
        repeat_policy="repeat_allowed",
        consecutive_repeat_policy="no_consecutive_repeat",
        terminal_status="terminal"
    ))

    engine.declare(TransformerStage(
        layer_index=layer_index,
        layer_name=layer_name,
        stage_name="softmax_activation_completed",
        status="computed"
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_softmax_activation_shape",
        previous_step_key="none",
        step_order=1,
        title="Softmax activation output shape",
        formula="softmax(z_i) = exp(z_i) / sum_j exp(z_j)",
        substitution=substitution,
        result=f"{output_shape} as probabilities"
    ))

    engine.declare(Layer(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape
    ))
