from facts import CalculationStep, LayerMetadata, TensorInfo, TrainingComponent, WeightShape


def declare_training_loss(
    engine,
    layer_index,
    layer_name,
    layer_title,
    loss_type,
    input_shape,
    label_shape,
    prediction_data_type,
    formula,
    substitution,
    reduction,
):
    output_shape = "scalar"
    engine.declare(TrainingComponent(
        component_index=layer_index,
        component_name=layer_name,
        component_title=layer_title,
        component_type="training_loss",
        input_shape=f"predictions: {input_shape}; labels: {label_shape}",
        output_shape=output_shape,
        status="computed",
        used_during="training_only",
    ))

    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="loss_parameters",
        shape="no trainable parameters",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name=f"{layer_name}_output",
        role="output",
        dims_count=0,
        shape=output_shape,
        data_type="scalar_loss",
        source=loss_type,
    ))

    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="training_component",
        layer_role="loss_function",
        repeat_policy="repeat_allowed",
        consecutive_repeat_policy="no_consecutive_loss",
        terminal_status="terminal",
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_loss",
        previous_step_key="none",
        step_order=1,
        title=layer_title,
        formula=formula,
        substitution=f"{substitution}; reduction={reduction}",
        result=output_shape,
    ))
