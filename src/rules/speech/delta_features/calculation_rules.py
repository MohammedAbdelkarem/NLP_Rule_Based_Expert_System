from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    ComponentGroup,
    LayerMetadata,
)
from src.rules.speech.delta_features.common.helpers import (
    build_delta_shapes,
    declare_delta_steps,
    declare_delta_tensors,
    delta_denominator,
)


class DeltaFeatureCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="delta_features",
            layer_title=MATCH.layer_title
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="mfcc_features",
            role="output",
            dims_count=4,
            shape=MATCH.previous_output_shape,
            data_type=MATCH.previous_data_type
        ),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="sample_rate",
                   role="shape_metadata", shape=MATCH.sample_rate),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_batch_size",
                   role="shape_metadata", shape=MATCH.batch_size),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_num_frames",
                   role="shape_metadata", shape=MATCH.num_frames),
        TensorInfo(layer_index=MATCH.previous_layer_index, tensor_name="output_mfcc_coefficients",
                   role="shape_metadata", shape=MATCH.num_mfcc_coefficients),
        TensorInfo(layer_index=MATCH.previous_layer_index,
                   tensor_name="output_channels", role="shape_metadata", shape=MATCH.channels),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            parameter_name="delta_window",
            value=MATCH.delta_window
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_delta_feature_streams(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        previous_data_type,
        sample_rate,
        batch_size,
        num_frames,
        num_mfcc_coefficients,
        channels,
        delta_window
    ):
        sample_rate_int = int(sample_rate)
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        num_mfcc_coefficients_int = int(num_mfcc_coefficients)
        channels_int = int(channels)
        delta_window_int = int(delta_window)

        shapes = build_delta_shapes(
            batch_size=batch_size_int,
            num_frames=num_frames_int,
            num_mfcc_coefficients=num_mfcc_coefficients_int,
            channels=channels_int,
        )
        denominator = delta_denominator(delta_window_int)

        declare_delta_tensors(
            self,
            layer_index,
            previous_layer_index,
            previous_output_shape,
            previous_data_type,
            sample_rate_int,
            batch_size_int,
            num_frames_int,
            num_mfcc_coefficients_int,
            shapes["concatenated_coefficients"],
            channels_int,
            delta_window_int,
            denominator,
            shapes["stream_shape"],
            shapes["concatenated_shape"],
        )

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="delta_features",
            group_title=layer_title,
            internal_layers=(
                "MFCC static stream preservation + fixed delta temporal-regression stream + "
                "fixed delta-delta stream + available feature-axis concatenation adapter"
            ),
            combine_operation="fixed_delta_and_delta_delta_temporal_feature_augmentation",
            input_shape=previous_output_shape,
            output_shape=shapes["separate_group_shape"],
            status="computed"
        ))
        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="delta_features",
            layer_kind="speech_feature_extraction_component_group",
            layer_role="separate_mfcc_static_delta_double_delta_temporal_feature_streams",
            repeat_policy="once_after_mfcc",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))
        declare_delta_steps(
            self,
            layer_index,
            delta_window_int,
            denominator,
            shapes["stream_shape"],
            num_mfcc_coefficients_int,
            shapes["concatenated_coefficients"],
            shapes["concatenated_shape"],
            shapes["separate_group_shape"],
        )
