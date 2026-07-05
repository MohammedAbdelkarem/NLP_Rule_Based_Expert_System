from src.facts import TensorInfo, WeightShape, CalculationStep


INPUT_STREAM_NAMES = (
    "mfcc_static",
    "delta",
    "delta_delta",
)

STREAM_SPECS = {
    "mfcc_static": {
        "stream_name": "mfcc_static",
        "tensor_name": "normalized_mfcc_static_stream",
        "data_type": "normalized_mfcc_static_stream_tensor",
    },
    "delta": {
        "stream_name": "delta",
        "tensor_name": "normalized_delta_stream",
        "data_type": "normalized_mfcc_delta_stream_tensor",
    },
    "delta_delta": {
        "stream_name": "delta_delta",
        "tensor_name": "normalized_delta_delta_stream",
        "data_type": "normalized_mfcc_delta_delta_stream_tensor",
    },
}


def normalization_mode_cmvn():
    return {
        "name": "cmvn_standardization",
        "display": "CMVN / standardization",
        "formula": "x_norm = (x - mean) / (std + epsilon)",
        "statistics": "per-stream mean and standard-deviation vectors from the training dataset",
        "statistic_a": "mean",
        "statistic_b": "std",
    }


def normalization_mode_range_scaling():
    return {
        "name": "range_scaling_minus_one_to_one",
        "display": "Range scaling to [-1, 1] with zero-centered statistics",
        "formula": "x_norm = 2 * (x - min) / (max - min + epsilon) - 1",
        "statistics": "per-stream min and max vectors from the training dataset",
        "statistic_a": "min",
        "statistic_b": "max",
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


def declare_normalization_streams(engine, layer_index, stream_shape, mode_info):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalized_mfcc_static_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="normalized_mfcc_static_stream_tensor",
        source=f"normalized_static_mfcc_features_using_{mode_info['name']}"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalized_delta_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="normalized_mfcc_delta_stream_tensor",
        source=f"normalized_delta_features_using_{mode_info['name']}"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalized_delta_delta_stream",
        role="stream_output",
        dims_count=4,
        shape=stream_shape,
        data_type="normalized_mfcc_delta_delta_stream_tensor",
        source=f"normalized_delta_delta_features_using_{mode_info['name']}"
    ))


def declare_normalization_statistics(engine, layer_index, statistics_shape, mode_info):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_a']}_mfcc_static",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_b']}_mfcc_static",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_a']}_delta",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_b']}_delta",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_a']}_delta_delta",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name=f"{mode_info['statistic_b']}_delta_delta",
        role="stored_statistics",
        dims_count=2,
        shape=statistics_shape,
        data_type="training_dataset_statistics",
        source="computed_from_training_split"
    ))


def declare_normalization_metadata(
    engine,
    layer_index,
    batch_size,
    num_frames,
    num_features,
    concatenated_coefficients,
    channels,
    mode_info,
):
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalization_mode",
        role="shape_metadata",
        dims_count=1,
        shape=mode_info["name"],
        data_type="configuration_value",
        source="user_selected_normalization_mode"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="output_batch_size",
        role="shape_metadata",
        dims_count=1,
        shape=str(batch_size),
        data_type="integer_dimension_value",
        source="feature_normalization_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="output_num_frames",
        role="shape_metadata",
        dims_count=1,
        shape=str(num_frames),
        data_type="integer_dimension_value",
        source="feature_normalization_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="output_features_per_stream",
        role="shape_metadata",
        dims_count=1,
        shape=str(num_features),
        data_type="integer_dimension_value",
        source="feature_normalization_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="concatenated_feature_coefficients",
        role="shape_metadata",
        dims_count=1,
        shape=str(concatenated_coefficients),
        data_type="integer_dimension_value",
        source="adapter_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="output_channels",
        role="shape_metadata",
        dims_count=1,
        shape=str(channels),
        data_type="integer_dimension_value",
        source="feature_normalization_shape_metadata"
    ))
    engine.declare(TensorInfo(
        layer_index=layer_index,
        layer_name="feature_normalization",
        tensor_name="normalization_adapter_available",
        role="shape_metadata",
        dims_count=1,
        shape="true",
        data_type="configuration_value",
        source="adapter_suggestion_metadata"
    ))


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
    declare_normalization_streams(engine, layer_index, stream_shape, mode_info)
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
        source="suggest_when_next_layer_requires_single_tensor"
    ))
    declare_normalization_statistics(
        engine, layer_index, statistics_shape, mode_info)
    declare_normalization_metadata(
        engine=engine,
        layer_index=layer_index,
        batch_size=batch_size,
        num_frames=num_frames,
        num_features=num_features,
        concatenated_coefficients=concatenated_coefficients,
        channels=channels,
        mode_info=mode_info,
    )


def declare_normalization_steps(
    engine,
    layer_index,
    mode_info,
    stream_shape,
    statistics_shape,
    separate_group_shape,
    adapter_shape,
):
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feature_normalization",
        step_key=f"layer_{layer_index}_normalization_mode",
        previous_step_key="none",
        step_order=1,
        title="Selected normalization mode",
        formula=mode_info["formula"],
        substitution=mode_info["display"],
        result=mode_info["name"]
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feature_normalization",
        step_key=f"layer_{layer_index}_statistics_shape",
        previous_step_key=f"layer_{layer_index}_normalization_mode",
        step_order=2,
        title="Stored statistics shape",
        formula="statistics are stored per stream and per channel",
        substitution=mode_info["statistics"],
        result=statistics_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feature_normalization",
        step_key=f"layer_{layer_index}_normalized_stream_shapes",
        previous_step_key=f"layer_{layer_index}_statistics_shape",
        step_order=3,
        title="Normalized stream shapes",
        formula="normalization preserves each stream shape",
        substitution=stream_shape,
        result=separate_group_shape
    ))
    engine.declare(CalculationStep(
        layer_index=layer_index,
        layer_name="feature_normalization",
        step_key=f"layer_{layer_index}_normalization_adapter",
        previous_step_key=f"layer_{layer_index}_normalized_stream_shapes",
        step_order=4,
        title="Adapter suggestion shape",
        formula="adapter concatenates normalized MFCC + Delta + Delta-Delta on the feature axis",
        substitution="use only if the next layer requires one tensor",
        result=adapter_shape
    ))
