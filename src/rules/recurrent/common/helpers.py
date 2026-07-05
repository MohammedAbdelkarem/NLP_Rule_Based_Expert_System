from src.facts import (
    CalculationStep,
    ComponentGroup,
    Layer,
    LayerMetadata,
    TensorInfo,
    WeightShape,
)


def shape_3d(batch_size, sequence_length, feature_dim):
    return f"({int(batch_size)}, {int(sequence_length)}, {int(feature_dim)})"


def shape_2d(batch_size, feature_dim):
    return f"({int(batch_size)}, {int(feature_dim)})"


def doubled(value):
    return int(value) * 2


def quadrupled(value):
    return int(value) * 4


def rnn_weight_summary(input_dim, hidden_size):
    return (
        f"W_xh: ({int(input_dim)}, {int(hidden_size)}), "
        f"W_hh: ({int(hidden_size)}, {int(hidden_size)}), "
        f"b_h: ({int(hidden_size)},)"
    )


def lstm_weight_summary(input_dim, hidden_size):
    four_hidden = quadrupled(hidden_size)
    return (
        f"W_x: ({int(input_dim)}, {four_hidden}), "
        f"W_h: ({int(hidden_size)}, {four_hidden}), "
        f"b: ({four_hidden},)"
    )


def stacked_rnn_weight_summary(input_dim, hidden_size, num_layers, unit_type):
    return (
        f"unit_type: {unit_type}; "
        f"layer_1_input_projection: ({int(input_dim)}, {int(hidden_size)}); "
        f"upper_recurrent_layers: {int(num_layers)} total layers; "
        f"hidden_to_hidden_core: ({int(hidden_size)}, {int(hidden_size)})"
    )


def bidirectional_weight_summary(input_dim, hidden_size, unit_type):
    return (
        f"unit_type: {unit_type}; "
        f"forward_cell_input: ({int(input_dim)}, {int(hidden_size)}); "
        f"backward_cell_input: ({int(input_dim)}, {int(hidden_size)}); "
        f"merged_output_dim: {doubled(hidden_size)}"
    )


def declare_recurrent_layer(
    engine,
    layer_index,
    layer_name,
    layer_title,
    input_shape,
    output_shape,
    output_dims_count,
    batch_size,
    sequence_length,
    input_dim,
    hidden_size,
    weight_name,
    parameter_shape,
    output_tensor_name,
    output_data_type,
    layer_role,
    return_mode,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="input_sequence_tensor",
        role="input",
        dims_count=3,
        shape=input_shape,
        data_type="float_sequence_tensor",
        source="user_or_previous_layer_sequence_output"
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name=weight_name,
        shape=parameter_shape
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name=output_tensor_name,
        role="output",
        dims_count=output_dims_count,
        shape=output_shape,
        data_type=output_data_type,
        source=f"{layer_name}_recurrent_transformation"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(batch_size)),
        data_type="integer_dimension_value",
        source="batch_size"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(sequence_length)),
        data_type="integer_dimension_value",
        source="sequence_length"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(hidden_size)),
        data_type="integer_dimension_value",
        source="hidden_size"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="input_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(input_dim)),
        data_type="integer_dimension_value",
        source="input_dim"
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="recurrent_layer",
        layer_role=layer_role,
        repeat_policy="allowed_with_new_hidden_size",
        consecutive_repeat_policy="allowed_when_returning_full_sequence",
        terminal_status="non_terminal"
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_input_shape",
        previous_step_key="sequence_input",
        step_order=1,
        title="Recurrent input sequence shape",
        formula="input = (batch_size, sequence_length, input_dim)",
        substitution=input_shape,
        result=input_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_weights",
        previous_step_key=f"layer_{layer_index}_{layer_name}_input_shape",
        step_order=2,
        title="Recurrent trainable parameter shapes",
        formula="input weights + recurrent hidden weights + bias",
        substitution=parameter_shape,
        result=parameter_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_{layer_name}_output_shape",
        previous_step_key=f"layer_{layer_index}_{layer_name}_weights",
        step_order=3,
        title="Recurrent output shape",
        formula=f"return_mode = {return_mode}",
        substitution=output_shape,
        result=output_shape
    ))
    engine.declare(Layer(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape
    ))


def declare_lstm_cell_state(engine, layer_index, batch_size, hidden_size):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="lstm",
        tensor_name="output_cell_state",
        role="state",
        dims_count=2,
        shape=shape_2d(batch_size, hidden_size),
        data_type="float_cell_state_tensor",
        source="lstm_internal_memory_state"
    ))


def declare_recurrent_component_group(
    engine,
    layer_index,
    group_name,
    group_title,
    input_shape,
    output_shape,
    output_dims_count,
    batch_size,
    sequence_length,
    input_dim,
    output_model_dim,
    hidden_size,
    parameter_shape,
    internal_layers,
    combine_operation,
    layer_role,
    return_mode,
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
        tensor_name="input_sequence_tensor",
        role="input",
        dims_count=3,
        shape=input_shape,
        data_type="float_sequence_tensor",
        source="user_or_previous_layer_sequence_output"
    ))
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=group_name,
        weight_name="recurrent_group_parameters",
        shape=parameter_shape
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="output_recurrent_tensor",
        role="output",
        dims_count=output_dims_count,
        shape=output_shape,
        data_type="float_recurrent_tensor",
        source=f"{group_name}_component_group_output"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(batch_size)),
        data_type="integer_dimension_value",
        source="batch_size"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="output_sequence_length",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(sequence_length)),
        data_type="integer_dimension_value",
        source="sequence_length"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="output_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(output_model_dim)),
        data_type="integer_dimension_value",
        source="recurrent_output_model_dim"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="hidden_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(hidden_size)),
        data_type="integer_dimension_value",
        source="hidden_size"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=group_name,
        tensor_name="input_model_dim",
        role="shape_metadata",
        dims_count=1,
        shape=str(int(input_dim)),
        data_type="integer_dimension_value",
        source="input_dim"
    ))
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=group_name,
        layer_kind="recurrent_component_group",
        layer_role=layer_role,
        repeat_policy="allowed_with_new_architecture",
        consecutive_repeat_policy="allowed_when_returning_full_sequence",
        terminal_status="non_terminal"
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=group_name,
        step_key=f"layer_{layer_index}_{group_name}_group_shape",
        previous_step_key="sequence_input",
        step_order=1,
        title="Recurrent component group output shape",
        formula=f"return_mode = {return_mode}",
        substitution=output_shape,
        result=output_shape
    ))
