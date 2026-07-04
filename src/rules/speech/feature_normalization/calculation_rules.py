from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    Layer,
    LayerMetadata,
)
from src.rules.speech.feature_normalization.common.helpers import (
    build_normalized_stream_shapes,
    declare_normalization_steps,
    declare_normalization_tensors,
    resolve_normalization_mode,
)


class FeatureNormalizationCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feature_normalization",
            layer_title=MATCH.layer_title
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="delta_feature_stream_group",
            role="output",
            dims_count=4,
            shape=MATCH.previous_output_shape,
            data_type="separate_delta_feature_stream_group"
        ),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_batch_size", role="shape_metadata", shape=MATCH.batch_size),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_num_frames", role="shape_metadata", shape=MATCH.num_frames),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="input_mfcc_coefficients", role="shape_metadata", shape=MATCH.num_features),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_channels", role="shape_metadata", shape=MATCH.channels),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode",
            value=MATCH.normalization_mode
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_feature_normalization(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        num_frames,
        num_features,
        channels,
        normalization_mode
    ):
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        num_features_int = int(num_features)
        channels_int = int(channels)
        mode_info = resolve_normalization_mode(normalization_mode)

        shapes = build_normalized_stream_shapes(
            batch_size=batch_size_int,
            num_frames=num_frames_int,
            num_features=num_features_int,
            channels=channels_int,
        )

        declare_normalization_tensors(
            engine=self,
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            input_shape=previous_output_shape,
            stream_shape=shapes["stream_shape"],
            separate_group_shape=shapes["separate_group_shape"],
            statistics_shape=shapes["statistics_shape"],
            adapter_shape=shapes["adapter_shape"],
            batch_size=batch_size_int,
            num_frames=num_frames_int,
            num_features=num_features_int,
            concatenated_coefficients=shapes["concatenated_coefficients"],
            channels=channels_int,
            mode_info=mode_info,
        )

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="feature_normalization",
            layer_title=layer_title,
            input_shape=previous_output_shape,
            output_shape=shapes["separate_group_shape"]
        ))
        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="feature_normalization",
            layer_kind="speech_feature_normalization_layer",
            layer_role="normalize_separated_mfcc_delta_delta_feature_streams",
            repeat_policy="once_after_delta_features",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        declare_normalization_steps(
            engine=self,
            layer_index=layer_index,
            mode_info=mode_info,
            stream_shape=shapes["stream_shape"],
            statistics_shape=shapes["statistics_shape"],
            separate_group_shape=shapes["separate_group_shape"],
            adapter_shape=shapes["adapter_shape"],
        )
