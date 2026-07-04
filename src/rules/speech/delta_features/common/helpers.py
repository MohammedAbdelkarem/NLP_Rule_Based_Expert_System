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
    squared_sum = sum(n * n for n in range(1, window_int + 1))
    return 2 * squared_sum


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

    stream_specs = [
        ("mfcc_static_stream", "mfcc_static_stream_tensor", "copied_static_mfcc_features"),
        ("delta_stream", "mfcc_delta_stream_tensor", f"fixed_temporal_regression_delta_window_{delta_window}"),
        ("delta_delta_stream", "mfcc_delta_delta_stream_tensor", f"fixed_temporal_regression_over_delta_window_{delta_window}"),
    ]
    for tensor_name, data_type, source in stream_specs:
        engine.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="delta_features",
            tensor_name=tensor_name,
            role="stream_output",
            dims_count=4,
            shape=stream_shape,
            data_type=data_type,
            source=source
        ))

    kernel_specs = [
        ("fixed_delta_temporal_kernel", f"window_radius={delta_window}, denominator={denominator}", "fixed_temporal_difference_kernel_not_trainable", "deterministic_delta_regression_formula"),
        ("fixed_delta_delta_temporal_kernel", f"window_radius={delta_window}, denominator={denominator}", "fixed_second_order_temporal_difference_kernel_not_trainable", "same_delta_formula_applied_to_delta_stream"),
    ]
    for tensor_name, shape, data_type, source in kernel_specs:
        engine.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="delta_features",
            tensor_name=tensor_name,
            role="intermediate",
            dims_count=1,
            shape=shape,
            data_type=data_type,
            source=source
        ))

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

    metadata = [
        ("sample_rate", sample_rate),
        ("output_batch_size", batch_size),
        ("output_num_frames", num_frames),
        ("input_mfcc_coefficients", num_mfcc_coefficients),
        ("concatenated_feature_coefficients", concatenated_coefficients),
        ("output_channels", channels),
        ("delta_window", delta_window),
        ("delta_adapter_available", "true"),
    ]
    for tensor_name, shape in metadata:
        engine.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="delta_features",
            tensor_name=tensor_name,
            role="shape_metadata",
            dims_count=1,
            shape=str(shape),
            data_type="integer_dimension_value" if str(shape).isdigit() else "configuration_value",
            source="delta_feature_shape_metadata"
        ))


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
    steps = [
        ("delta_denominator", "Fixed delta-regression denominator", "2 * sum(n^2 for n = 1..N)", f"N={delta_window}, denominator={denominator}", str(denominator)),
        ("delta_stream_shapes", "Static, delta, and delta-delta stream shapes", "each stream keeps (batch_size, num_frames, num_mfcc_coefficients, channels)", stream_shape, f"mfcc_static={stream_shape}, delta={stream_shape}, delta_delta={stream_shape}"),
        ("delta_adapter_shape", "Available concatenation adapter shape", "num_mfcc_coefficients * 3", f"{num_mfcc_coefficients} * 3 = {concatenated_coefficients}", concatenated_shape),
        ("delta_primary_output_shape", "Primary output remains separated", "output = separate MFCC, Delta, Delta-Delta streams", "output_design=separate_streams_always", separate_group_shape),
        ("delta_parameter_status", "Trainable parameter status", "delta features use fixed temporal difference kernels", f"delta_window={delta_window}", "trainable=False, has_weights=False, has_bias=False, has_kernel=True, kernel_trainable=False"),
    ]
    previous_key = "none"
    for order, (key, title, formula, substitution, result) in enumerate(steps, start=1):
        step_key = f"layer_{layer_index}_{key}"
        engine.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="delta_features",
            step_key=step_key,
            previous_step_key=previous_key,
            step_order=order,
            title=title,
            formula=formula,
            substitution=substitution,
            result=result
        ))
        previous_key = step_key
