from facts import (
    CalculationStep,
    ComponentGroup,
    IntermediateTensor,
    Layer,
    LayerMetadata,
    TensorInfo,
    TransformerStage,
    WeightShape,
)


def classifier_parameter_shape(input_dim, hidden_dim, output_dim):
    return (
        f"W_hidden: ({input_dim}, {hidden_dim}), "
        f"b_hidden: ({hidden_dim},), "
        f"W_out: ({hidden_dim}, {output_dim}), "
        f"b_out: ({output_dim},)"
    )


def classifier_shape(batch_size, feature_dim):
    return f"({batch_size}, {feature_dim})"


def declare_classifier_metadata(
    engine,
    layer_index,
    batch_size,
    output_dim,
    classifier_mode,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source="classifier_input_batch_size",
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(output_dim),
        data_type="integer_dimension_value",
        source="classifier_output_dimension",
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="classifier_mode",
        role="configuration",
        dims_count=1,
        shape=classifier_mode,
        data_type="configuration_value",
        source="user_selected_classifier_mode",
    ))


def declare_classifier_intermediates(
    engine,
    layer_index,
    hidden_shape,
    logits_shape,
    probability_shape,
    probability_source,
):
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="hidden_projection_output",
        shape=hidden_shape,
        data_type="float_hidden_tensor",
        source="trainable_hidden_projection",
    ))
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="activation_output",
        shape=hidden_shape,
        data_type="float_hidden_tensor",
        source="hidden_nonlinear_activation",
    ))
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="logits_output",
        shape=logits_shape,
        data_type="logits",
        source="trainable_output_projection",
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        tensor_name="classifier_probability_output",
        role="output",
        dims_count=2,
        shape=probability_shape,
        data_type="probabilities",
        source=probability_source,
    ))


def declare_classifier_common(
    engine,
    layer_index,
    layer_title,
    input_shape,
    hidden_shape,
    logits_shape,
    probability_shape,
    input_dim,
    hidden_dim,
    output_dim,
    batch_size,
    classifier_mode,
    probability_activation,
):
    parameter_shape = classifier_parameter_shape(input_dim, hidden_dim, output_dim)
    internal_layers = (
        "Hidden dense projection + ReLU activation + "
        f"output projection + {probability_activation}"
    )
    combine_operation = (
        "The hidden projection learns features, the output projection produces logits, "
        f"and {probability_activation} converts logits into probabilities."
    )
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        weight_name="feedforward_classifier_parameters",
        shape=parameter_shape,
    ))
    declare_classifier_intermediates(
        engine=engine,
        layer_index=layer_index,
        hidden_shape=hidden_shape,
        logits_shape=logits_shape,
        probability_shape=probability_shape,
        probability_source=f"{probability_activation}_probability_activation",
    )
    declare_classifier_metadata(
        engine=engine,
        layer_index=layer_index,
        batch_size=batch_size,
        output_dim=output_dim,
        classifier_mode=classifier_mode,
    )
    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name="feedforward_classifier_head",
        group_title=layer_title,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
        input_shape=input_shape,
        output_shape=probability_shape,
        status="computed",
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        layer_kind="classifier_head",
        layer_role="classification_output_head",
        repeat_policy="single_terminal_head",
        consecutive_repeat_policy="no_consecutive_repeat",
        terminal_status="terminal",
    ))
    engine.declare(TransformerStage(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        stage_name="feedforward_classifier_head_completed",
        status="computed",
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        step_key=f"layer_{layer_index}_classifier_hidden_projection_shape",
        previous_step_key="none",
        step_order=1,
        title="Classifier hidden projection shape",
        formula="H = XW_hidden + b_hidden",
        substitution=f"input_dim={input_dim}, hidden_dim={hidden_dim}",
        result=hidden_shape,
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        step_key=f"layer_{layer_index}_classifier_logits_shape",
        previous_step_key=f"layer_{layer_index}_classifier_hidden_projection_shape",
        step_order=2,
        title="Classifier logits shape",
        formula="Z = ReLU(H)W_out + b_out",
        substitution=f"hidden_dim={hidden_dim}, output_dim={output_dim}",
        result=logits_shape,
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        step_key=f"layer_{layer_index}_classifier_probability_shape",
        previous_step_key=f"layer_{layer_index}_classifier_logits_shape",
        step_order=3,
        title="Classifier probability shape",
        formula=f"Y = {probability_activation}(Z)",
        substitution=classifier_mode,
        result=probability_shape,
    ))
    engine.declare(Layer(
        layer_index=layer_index,
        layer_name="feedforward_classifier_head",
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=probability_shape,
    ))
