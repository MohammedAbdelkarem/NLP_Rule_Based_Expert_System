from src.facts import TensorInfo, WeightShape, CalculationStep


STREAM_SPECS = [
    ("normalized_mfcc_static_stream", "normalized_mfcc_static_stream_tensor", "normalized_static_mfcc_features"),
    ("normalized_delta_stream", "normalized_mfcc_delta_stream_tensor", "normalized_delta_features"),
    ("normalized_delta_delta_stream", "normalized_mfcc_delta_delta_stream_tensor", "normalized_delta_delta_features"),
]


INPUT_STREAM_NAMES = [
    "mfcc_static_stream",
    "delta_stream",
    "delta_delta_stream",
]


def resolve_normalization_mode(mode_choice):
    choice = str(mode_choice)
    if choice == "2":
        return {
            "name": "range_scaling_minus_one_to_one",
            "display": "Range scaling to [-1, 1] with zero-centered statistics",
            "formula": "x_norm = 2 * (x - min) / (max - min + epsilon) - 1",
            "statistics": "per-stream min and max vectors from the training dataset",
            "statistics_names": ("min", "max"),
        }

    return {
        "name": "cmvn_standardization",
        "display": "CMVN / standardization",
        "formula": "x_norm = (x - mean) / (std + epsilon)",
        "statistics": "per-stream mean and standard-deviation vectors from the training dataset",
        "statistics_names": ("mean", "std"),
    }


def build_normalized_stream_shapes(batch_size, num_frames, num_features, channels):
    stream_shape = f"({batch_size}, {num_frames}, {num_features}, {channels})"
    separate_group_shape = (
        f"normalized_mfcc={stream_shape}; "
        f"normalized_delta={stream_shape}; "
        f"normalized_delta_delta={stream_shape}"
    )
    statistics_shape = f"({num_features}, {channels})"
    concatenated_coefficients = int(num_features) * 3
    adapter_shape = f"({batch_size}, {num_frames}, {concatenated_coefficients}, {channels})"
    return {
        "stream_shape": stream_shape,
        "separate_group_shape": separate_group_shape,
        "statistics_shape": statistics_shape,
        "adapter_shape": adapter_shape,
        "concatenated_coefficients": concatenated_coefficients,
    }


def declare_normalization_tensors(
    engine,
    layer_index,
    previous_layer_index,
    input_shape,
    stream_shape,
    separate_group_shape,
    statistics_shape,
    adapter_shape,
    batch_size,
    num_frames,
    num_features,
    concatenated_coefficients,
    channels,
    mode_info,
):
    engine.declare(WeightShape(
        layer_index=layer_index,
        layer_name="feature_normalization",
        weight_name="feature_normalization_parameters",
        shape="None (dataset statistics are stored; not trainable weights)"
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="input_delta_feature_stream_group",
        role="input",
        dims_count=4,
        shape=input_shape,
        data_type="separate_delta_feature_stream_group",
        source=f"layer_{previous_layer_index}_delta_feature_stream_group"
    ))

    for tensor_name, data_type, source in STREAM_SPECS:
        engine.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feature_normalization",
            tensor_name=tensor_name,
            role="stream_output",
            dims_count=4,
            shape=stream_shape,
            data_type=data_type,
            source=f"{source}_using_{mode_info['name']}"
        ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalized_delta_feature_stream_group",
        role="output",
        dims_count=4,
        shape=separate_group_shape,
        data_type="normalized_separate_delta_feature_stream_group",
        source="feature_normalization_primary_output_separate_streams"
    ))

    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalized_delta_feature_concatenation_adapter",
        role="adapter_suggestion",
        dims_count=4,
        shape=adapter_shape,
        data_type="normalized_mfcc_delta_delta_concatenated_tensor",
        source="suggest_when_next_layer_requires_single_concatenated_feature_tensor"
    ))

    stat_a, stat_b = mode_info["statistics_names"]
    for stream_name in ("mfcc", "delta", "delta_delta"):
        for stat_name in (stat_a, stat_b):
            engine.declare(TensorInfo(
                layer_index=layer_index,
                layer_name="feature_normalization",
                tensor_name=f"{stream_name}_{stat_name}_statistics",
                role="stored_statistics",
                dims_count=2,
                shape=statistics_shape,
                data_type="fixed_training_dataset_statistics_not_trainable",
                source=f"training_dataset_{stream_name}_{stat_name}"
            ))

    metadata = [
        ("output_batch_size", batch_size),
        ("output_num_frames", num_frames),
        ("output_feature_coefficients", num_features),
        ("concatenated_feature_coefficients", concatenated_coefficients),
        ("output_channels", channels),
        ("normalization_mode", mode_info["name"]),
        ("statistics_shape", statistics_shape),
        ("adapter_available", "true"),
    ]
    for tensor_name, shape in metadata:
        engine.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="feature_normalization",
            tensor_name=tensor_name,
            role="shape_metadata",
            dims_count=1,
            shape=str(shape),
            data_type="integer_dimension_value" if str(shape).isdigit() else "configuration_value",
            source="feature_normalization_shape_metadata"
        ))


def declare_normalization_steps(
    engine,
    layer_index,
    mode_info,
    stream_shape,
    statistics_shape,
    separate_group_shape,
    adapter_shape,
):
    steps = [
        ("normalization_mode", "Selected feature normalization mode", mode_info["formula"], mode_info["display"], mode_info["statistics"]),
        ("statistics_shape", "Stored statistics shape", "statistics are stored per feature and channel for each stream", statistics_shape, "trainable=False, has_weights=False, has_statistics=True"),
        ("stream_shapes", "Normalized stream shapes", "normalization preserves each stream shape", stream_shape, separate_group_shape),
        ("adapter_suggestion_shape", "Available concatenation adapter shape", "three normalized streams can be concatenated on the feature axis when needed", adapter_shape, "suggest only when the selected next layer requires one tensor"),
    ]

    previous_key = "none"
    for order, (key, title, formula, substitution, result) in enumerate(steps, start=1):
        step_key = f"layer_{layer_index}_{key}"
        engine.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="feature_normalization",
            step_key=step_key,
            previous_step_key=previous_key,
            step_order=order,
            title=title,
            formula=formula,
            substitution=substitution,
            result=result
        ))
        previous_key = step_key
