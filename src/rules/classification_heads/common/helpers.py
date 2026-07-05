from src.facts import (
    ComponentGroup,
    CalculationStep,
    IntermediateTensor,
    LayerMetadata,
    TensorInfo,
    WeightShape,
)


def make_shape(*dims):
    return f"({', '.join(str(dim) for dim in dims)})"


def make_rank_2_shape(batch_size, feature_dim):
    return f"({batch_size}, {feature_dim})"


def make_rank_3_shape(batch_size, feature_dim, sequence_length):
    return f"({batch_size}, {sequence_length}, {feature_dim})"


def make_rank_2_class_index_label_shape(batch_size):
    return f"({batch_size},)"


def make_rank_3_class_index_label_shape(batch_size, sequence_length):
    return f"({batch_size}, {sequence_length})"


def make_label_shape(batch_size, sequence_length=None):
    if sequence_length is None:
        return make_rank_2_class_index_label_shape(batch_size)
    return make_rank_3_class_index_label_shape(batch_size, sequence_length)


def declare_output_shape_metadata(
    engine,
    layer_index,
    layer_name,
    batch_size,
    output_dim,
    batch_source,
    output_dim_source,
    sequence_length=None,
    sequence_source=None,
):
    if sequence_length is None:
        declare_output_shape_metadata_rank_2(
            engine=engine,
            layer_index=layer_index,
            layer_name=layer_name,
            batch_size=batch_size,
            output_dim=output_dim,
            batch_source=batch_source,
            output_dim_source=output_dim_source,
        )
        return

    declare_output_shape_metadata_rank_3(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=output_dim,
        sequence_length=sequence_length,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
        sequence_source=sequence_source,
    )


def declare_logistic_head(
    engine,
    layer_index,
    layer_name,
    layer_title,
    input_shape,
    output_shape,
    input_dim,
    batch_size,
    output_dim=1,
    sequence_length=None,
    batch_source="classifier_input_batch_size",
    output_dim_source="classifier_output_dimension",
    sequence_source="classifier_input_sequence_length",
):
    if sequence_length is None:
        declare_binary_logistic_head_rank_2(
            engine=engine,
            layer_index=layer_index,
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape,
            input_dim=input_dim,
            batch_size=batch_size,
            batch_source=batch_source,
            output_dim_source=output_dim_source,
        )
        return

    declare_binary_logistic_head_rank_3(
        engine=engine,
        layer_index=layer_index,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
        input_dim=input_dim,
        batch_size=batch_size,
        sequence_length=sequence_length,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
        sequence_source=sequence_source,
    )


def declare_output_shape_metadata_rank_2(
    engine,
    layer_index,
    layer_name,
    batch_size,
    output_dim,
    batch_source,
    output_dim_source,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source=batch_source,
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(output_dim),
        data_type="integer_dimension_value",
        source=output_dim_source,
    ))


def declare_output_shape_metadata_rank_3(
    engine,
    layer_index,
    layer_name,
    batch_size,
    output_dim,
    sequence_length,
    batch_source,
    output_dim_source,
    sequence_source,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source=batch_source,
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(sequence_length),
        data_type="integer_dimension_value",
        source=sequence_source,
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(output_dim),
        data_type="integer_dimension_value",
        source=output_dim_source,
    ))


def declare_logistic_head_base(
    engine,
    layer_index,
    layer_name,
    layer_title,
    input_shape,
    output_shape,
    logits_shape,
    input_dim,
    output_dim,
    output_data_type,
    dims_count,
    activation_name,
    probability_formula,
    probability_result,
):
    internal_layers = f"linear_logits_projection + {activation_name}"
    combine_operation = "linear projection followed by probability activation"
    parameter_shape = f"W: ({input_dim}, {output_dim})\nb: ({output_dim},)"

    engine.declare(ComponentGroup(
        group_index=layer_index,
        group_name=layer_name,
        group_title=layer_title,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
        input_shape=input_shape,
        output_shape=output_shape,
        status="computed",
    ))

    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="classifier_parameters",
        shape=parameter_shape,
    ))

    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="linear_logits",
        shape=logits_shape,
        data_type="logits",
        source="linear_logits_projection",
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name=f"{layer_name}_output",
        role="output",
        dims_count=dims_count,
        shape=output_shape,
        data_type=output_data_type,
        source=f"{activation_name}_probability_activation",
    ))

    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="component_group",
        layer_role="classification_head",
        repeat_policy="repeat_allowed",
        consecutive_repeat_policy="no_consecutive_classification_head",
        terminal_status="terminal",
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_linear_logits",
        previous_step_key="none",
        step_order=1,
        title="Linear logits projection",
        formula="z = xW + b",
        substitution=f"{input_shape} × ({input_dim}, {output_dim}) + ({output_dim},)",
        result=logits_shape,
    ))

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_probabilities",
        previous_step_key=f"layer_{layer_index}_{layer_name}_linear_logits",
        step_order=2,
        title=f"{activation_name.capitalize()} probability activation",
        formula=probability_formula,
        substitution=f"logits shape {logits_shape}",
        result=probability_result,
    ))


def declare_binary_logistic_head_rank_2(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    input_dim,
    batch_size,
    batch_source="classifier_input_batch_size",
    output_dim_source="classifier_output_dimension",
):
    layer_name = "binary_logistic_regression_head"
    declare_logistic_head_base(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
        logits_shape=output_shape,
        input_dim=input_dim,
        output_dim=1,
        output_data_type="binary_probability",
        dims_count=2,
        activation_name="sigmoid",
        probability_formula="p = sigmoid(xW + b)",
        probability_result=f"{output_shape} as binary probabilities",
    )
    declare_output_shape_metadata_rank_2(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=1,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
    )


def declare_binary_logistic_head_rank_3(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    input_dim,
    batch_size,
    sequence_length,
    batch_source="classifier_input_batch_size",
    sequence_source="classifier_input_sequence_length",
    output_dim_source="classifier_output_dimension",
):
    layer_name = "binary_logistic_regression_head"
    declare_logistic_head_base(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
        logits_shape=output_shape,
        input_dim=input_dim,
        output_dim=1,
        output_data_type="binary_probability",
        dims_count=3,
        activation_name="sigmoid",
        probability_formula="p = sigmoid(xW + b)",
        probability_result=f"{output_shape} as binary probabilities",
    )
    declare_output_shape_metadata_rank_3(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=1,
        sequence_length=sequence_length,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
        sequence_source=sequence_source,
    )


def declare_multinomial_logistic_head_rank_2(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    input_dim,
    num_classes,
    batch_size,
    batch_source="classifier_input_batch_size",
    output_dim_source="classifier_output_dimension",
):
    layer_name = "multinomial_logistic_regression_head"
    declare_logistic_head_base(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
        logits_shape=output_shape,
        input_dim=input_dim,
        output_dim=num_classes,
        output_data_type="class_probabilities",
        dims_count=2,
        activation_name="softmax",
        probability_formula="p = softmax(xW + b)",
        probability_result=f"{output_shape} as class probabilities",
    )
    declare_output_shape_metadata_rank_2(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=num_classes,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
    )


def declare_multinomial_logistic_head_rank_3(
    engine,
    layer_index,
    layer_title,
    input_shape,
    output_shape,
    input_dim,
    num_classes,
    batch_size,
    sequence_length,
    batch_source="classifier_input_batch_size",
    sequence_source="classifier_input_sequence_length",
    output_dim_source="classifier_output_dimension",
):
    layer_name = "multinomial_logistic_regression_head"
    declare_logistic_head_base(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
        logits_shape=output_shape,
        input_dim=input_dim,
        output_dim=num_classes,
        output_data_type="class_probabilities",
        dims_count=3,
        activation_name="softmax",
        probability_formula="p = softmax(xW + b)",
        probability_result=f"{output_shape} as class probabilities",
    )
    declare_output_shape_metadata_rank_3(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=num_classes,
        sequence_length=sequence_length,
        batch_source=batch_source,
        output_dim_source=output_dim_source,
        sequence_source=sequence_source,
    )
