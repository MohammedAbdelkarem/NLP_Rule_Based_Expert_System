from src.facts import TensorInfo, WeightShape, CalculationStep


def build_delta_shapes(batch_size, num_frames, num_mfcc_coefficients, channels):
    stream_shape = f"({batch_size}, {num_frames}, {num_mfcc_coefficients}, {channels})"
    concatenated_coefficients = int(num_mfcc_coefficients) * 3
    concatenated_shape = f"({batch_size}, {num_frames}, {concatenated_coefficients}, {channels})"
    separate_group_shape = (
        f"mfcc={stream_shape}; "
        f"delta={stream_shape}; "
        f"delta_delta={stream_shape}"
    )
    return {
        "stream_shape": stream_shape,
        "separate_group_shape": separate_group_shape,
        "concatenated_coefficients": concatenated_coefficients,
        "concatenated_shape": concatenated_shape,
    }


def delta_denominator(delta_window):
    window_int = int(delta_window)
    squared_sum = window_int * (window_int + 1) * ((2 * window_int) + 1) // 6
    return 2 * squared_sum


def declare_delta_stream_tensors(engine, layer_index, stream_shape, delta_window):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="mfcc_static_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="mfcc_static_stream_tensor",
        source="copied_static_mfcc_features"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="delta_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="mfcc_delta_stream_tensor",
        source=f"fixed_temporal_regression_delta_window_{delta_window}"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="delta_delta_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="mfcc_delta_delta_stream_tensor",
        source=f"fixed_temporal_regression_over_delta_window_{delta_window}"
    ))


def declare_delta_kernel_tensors(engine, layer_index, delta_window, denominator):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="fixed_delta_temporal_kernel",
        role="intermediate",
        dims_count=1,
        shape=f"window_radius={delta_window}, denominator={denominator}",
        data_type="fixed_temporal_difference_kernel_not_trainable",
        source="deterministic_delta_regression_formula"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="fixed_delta_delta_temporal_kernel",
        role="intermediate",
        dims_count=1,
        shape=f"window_radius={delta_window}, denominator={denominator}",
        data_type="fixed_second_order_temporal_difference_kernel_not_trainable",
        source="same_delta_formula_applied_to_delta_stream"
    ))


def declare_delta_metadata(
    engine,
    layer_index,
    sample_rate,
    batch_size,
    num_frames,
    num_mfcc_coefficients,
    concatenated_coefficients,
    channels,
    delta_window,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="sample_rate",
        role="shape_metadata",
        dims_count=1,
        shape=str(sample_rate),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="output_num_frames",
        role="shape_metadata",
        dims_count=1,
        shape=str(num_frames),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="input_mfcc_coefficients",
        role="shape_metadata",
        dims_count=1,
        shape=str(num_mfcc_coefficients),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="concatenated_feature_coefficients",
        role="shape_metadata",
        dims_count=1,
        shape=str(concatenated_coefficients),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="output_channels",
        role="shape_metadata",
        dims_count=1,
        shape=str(channels),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="delta_window",
        role="shape_metadata",
        dims_count=1,
        shape=str(delta_window),
        data_type="integer_dimension_value",
        source="delta_feature_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="delta_adapter_available",
        role="shape_metadata",
        dims_count=1,
        shape="true",
        data_type="configuration_value",
        source="delta_feature_shape_metadata"
    ))


def declare_delta_tensors(
    engine,
    layer_index,
    previous_layer_index,
    input_shape,
    previous_data_type,
    sample_rate,
    batch_size,
    num_frames,
    num_mfcc_coefficients,
    concatenated_coefficients,
    channels,
    delta_window,
    denominator,
    stream_shape,
    concatenated_shape,
):
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="delta_features",
        weight_name="delta_feature_parameters",
        shape="None (fixed temporal-difference kernels; not trainable)"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="input_mfcc_features",
        role="input",
        dims_count=4,
        shape=input_shape,
        data_type=previous_data_type,
        source=f"layer_{previous_layer_index}_mfcc_features"
    ))
    declare_delta_stream_tensors(engine, layer_index, stream_shape, delta_window)
    declare_delta_kernel_tensors(engine, layer_index, delta_window, denominator)
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="delta_features",
        tensor_name="delta_feature_concatenation_adapter",
        role="adapter_output",
        dims_count=4,
        shape=concatenated_shape,
        data_type="mfcc_delta_delta_concatenated_tensor",
        source="concatenate_mfcc_delta_delta_streams_on_feature_axis"
    ))
    declare_delta_metadata(
        engine=engine,
        layer_index=layer_index,
        sample_rate=sample_rate,
        batch_size=batch_size,
        num_frames=num_frames,
        num_mfcc_coefficients=num_mfcc_coefficients,
        concatenated_coefficients=concatenated_coefficients,
        channels=channels,
        delta_window=delta_window,
    )


def declare_delta_steps(
    engine,
    layer_index,
    delta_window,
    denominator,
    stream_shape,
    num_mfcc_coefficients,
    concatenated_coefficients,
    concatenated_shape,
    separate_group_shape,
):
    step_1_key = f"layer_{layer_index}_delta_denominator"
    step_2_key = f"layer_{layer_index}_delta_stream_shapes"
    step_3_key = f"layer_{layer_index}_delta_adapter_shape"
    step_4_key = f"layer_{layer_index}_delta_primary_output_shape"
    step_5_key = f"layer_{layer_index}_delta_parameter_status"

    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="delta_features",
        step_key=step_1_key,
        previous_step_key="none",
        step_order=1,
        title="Fixed delta-regression denominator",
        formula="2 * N * (N + 1) * (2N + 1) / 6",
        substitution=f"N={delta_window}, denominator={denominator}",
        result=str(denominator)
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="delta_features",
        step_key=step_2_key,
        previous_step_key=step_1_key,
        step_order=2,
        title="Static, delta, and delta-delta stream shapes",
        formula="each stream keeps (batch_size, num_frames, num_mfcc_coefficients, channels)",
        substitution=stream_shape,
        result=f"mfcc_static={stream_shape}, delta={stream_shape}, delta_delta={stream_shape}"
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="delta_features",
        step_key=step_3_key,
        previous_step_key=step_2_key,
        step_order=3,
        title="Available concatenation adapter shape",
        formula="num_mfcc_coefficients * 3",
        substitution=f"{num_mfcc_coefficients} * 3 = {concatenated_coefficients}",
        result=concatenated_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="delta_features",
        step_key=step_4_key,
        previous_step_key=step_3_key,
        step_order=4,
        title="Primary output remains separated",
        formula="output = separate MFCC, Delta, Delta-Delta streams",
        substitution="output_design=separate_streams_always",
        result=separate_group_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="delta_features",
        step_key=step_5_key,
        previous_step_key=step_4_key,
        step_order=5,
        title="Trainable parameter status",
        formula="delta features use fixed temporal difference kernels",
        substitution=f"delta_window={delta_window}",
        result="trainable=False, has_weights=False, has_bias=False, has_kernel=True, kernel_trainable=False"
    ))
