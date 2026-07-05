from src.facts import (
    CalculationStep,
    ComponentGroup,
    IntermediateTensor,
    Layer,
    LayerMetadata,
    TensorInfo,
    TransformerStage,
    WeightShape,
)


def feature_vector_shape(batch_size, feature_dim):
    return f"({batch_size}, {feature_dim})"


def sequence_tensor_shape(batch_size, sequence_length, feature_dim):
    return f"({batch_size}, {sequence_length}, {feature_dim})"


def dense_parameter_shape(input_dim, output_dim):
    return f"W: ({input_dim}, {output_dim}), b: ({output_dim},)"


def mlp_parameter_shape(input_dim, hidden_dim, output_dim):
    return (
        f"W1: ({input_dim}, {hidden_dim}), "
        f"b1: ({hidden_dim},), "
        f"W2: ({hidden_dim}, {output_dim}), "
        f"b2: ({output_dim},)"
    )


def declare_rank_2_shape_metadata(
    engine,
    layer_index,
    layer_name,
    batch_size,
    output_dim,
    batch_source,
    output_source,
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
        source=output_source,
    ))


def declare_rank_3_shape_metadata(
    engine,
    layer_index,
    layer_name,
    batch_size,
    sequence_length,
    output_dim,
    batch_source,
    sequence_source,
    output_source,
):
    declare_rank_2_shape_metadata(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        batch_size=batch_size,
        output_dim=output_dim,
        batch_source=batch_source,
        output_source=output_source,
    )
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


def declare_grouped_layer_output(
    engine,
    layer_index,
    layer_name,
    layer_title,
    layer_role,
    input_shape,
    output_shape,
    internal_layers,
    combine_operation,
):
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
    engine.declare(LayerMetadata(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_kind="grouped_layer",
        layer_role=layer_role,
        repeat_policy="repeat_allowed",
        consecutive_repeat_policy="allow_consecutive_repeat",
        terminal_status="non_terminal",
    ))
    engine.declare(TransformerStage(
        layer_index=layer_index,
        layer_name=layer_name,
        stage_name=f"{layer_name}_completed",
        status="computed",
    ))
    engine.declare(Layer(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        input_shape=input_shape,
        output_shape=output_shape,
    ))


def declare_hidden_dense_intermediates(
    engine,
    layer_index,
    layer_name,
    projection_shape,
    activation_shape,
    activation_name,
):
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="dense_projection_output",
        shape=projection_shape,
        data_type="float_hidden_tensor",
        source="trainable_dense_projection_W_b",
    ))
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="activation_output",
        shape=activation_shape,
        data_type="float_hidden_tensor",
        source=f"{activation_name}_non_linear_activation",
    ))


def declare_hidden_dense_common(
    engine,
    layer_index,
    layer_title,
    dims_count,
    input_shape,
    output_shape,
    input_dim,
    hidden_dim,
    activation_name,
):
    layer_name = "hidden_dense_activation_block"
    parameter_shape = dense_parameter_shape(input_dim, hidden_dim)
    internal_layers = f"Dense projection + {activation_name.upper()} activation"
    combine_operation = (
        "The dense projection changes the feature dimension, "
        "then the activation preserves the projected shape."
    )
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="hidden_dense_parameters",
        shape=parameter_shape,
    ))
    declare_hidden_dense_intermediates(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        projection_shape=output_shape,
        activation_shape=output_shape,
        activation_name=activation_name,
    )
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="hidden_dense_activation_output",
        role="output",
        dims_count=dims_count,
        shape=output_shape,
        data_type="float_hidden_tensor",
        source="hidden_dense_activation_block_output",
    ))
    declare_grouped_layer_output(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        layer_role="hidden_feature_transformation",
        input_shape=input_shape,
        output_shape=output_shape,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
    )
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_hidden_dense_projection_shape",
        previous_step_key="none",
        step_order=1,
        title="Dense projection shape",
        formula="Z = XW + b",
        substitution=f"feature_dim={input_dim}, hidden_dim={hidden_dim}",
        result=output_shape,
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_hidden_dense_activation_shape",
        previous_step_key=f"layer_{layer_index}_hidden_dense_projection_shape",
        step_order=2,
        title="Activation output shape",
        formula=f"A = {activation_name}(Z)",
        substitution="activation is element-wise",
        result=output_shape,
    ))


def declare_mlp_intermediates(
    engine,
    layer_index,
    layer_name,
    first_projection_shape,
    activation_shape,
    output_shape,
    activation_name,
):
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="first_projection_output",
        shape=first_projection_shape,
        data_type="float_hidden_tensor",
        source="trainable_projection_W1_b1",
    ))
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="activation_output",
        shape=activation_shape,
        data_type="float_hidden_tensor",
        source=f"{activation_name}_non_linear_activation",
    ))
    engine.declare(IntermediateTensor(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="second_projection_output",
        shape=output_shape,
        data_type="float_hidden_tensor",
        source="trainable_projection_W2_b2",
    ))


def declare_mlp_common(
    engine,
    layer_index,
    layer_title,
    dims_count,
    input_shape,
    hidden_shape,
    output_shape,
    input_dim,
    hidden_dim,
    output_dim,
    activation_name,
):
    layer_name = "feedforward_mlp_block"
    parameter_shape = mlp_parameter_shape(input_dim, hidden_dim, output_dim)
    internal_layers = (
        f"Linear projection W1 + {activation_name.upper()} activation + "
        "Linear projection W2"
    )
    combine_operation = (
        "The first projection maps input_dim to hidden_dim, "
        "the activation preserves hidden_dim, and the second projection "
        "maps hidden_dim to output_dim."
    )
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name=layer_name,
        weight_name="feedforward_mlp_parameters",
        shape=parameter_shape,
    ))
    declare_mlp_intermediates(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        first_projection_shape=hidden_shape,
        activation_shape=hidden_shape,
        output_shape=output_shape,
        activation_name=activation_name,
    )
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name=layer_name,
        tensor_name="feedforward_mlp_output",
        role="output",
        dims_count=dims_count,
        shape=output_shape,
        data_type="float_hidden_tensor",
        source="feedforward_mlp_block_output",
    ))
    declare_grouped_layer_output(
        engine=engine,
        layer_index=layer_index,
        layer_name=layer_name,
        layer_title=layer_title,
        layer_role="feedforward_feature_transformation",
        input_shape=input_shape,
        output_shape=output_shape,
        internal_layers=internal_layers,
        combine_operation=combine_operation,
    )
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_mlp_first_projection_shape",
        previous_step_key="none",
        step_order=1,
        title="MLP first projection shape",
        formula="H = XW1 + b1",
        substitution=f"input_dim={input_dim}, hidden_dim={hidden_dim}",
        result=hidden_shape,
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_mlp_activation_shape",
        previous_step_key=f"layer_{layer_index}_mlp_first_projection_shape",
        step_order=2,
        title="MLP activation shape",
        formula=f"A = {activation_name}(H)",
        substitution="activation is element-wise",
        result=hidden_shape,
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name=layer_name,
        step_key=f"layer_{layer_index}_mlp_second_projection_shape",
        previous_step_key=f"layer_{layer_index}_mlp_activation_shape",
        step_order=3,
        title="MLP second projection shape",
        formula="Y = AW2 + b2",
        substitution=f"hidden_dim={hidden_dim}, output_dim={output_dim}",
        result=output_shape,
    ))
