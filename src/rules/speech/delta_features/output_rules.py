from experta import Rule, MATCH, NOT

from src.facts import (
    Layer,
    ComponentGroup,
    LayerParameter,
    TensorInfo,
    WeightShape,
    OutputStatus,
)
from src.utils.print_helpers import print_delta_features_result


class DeltaFeatureOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name="delta_features",
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            status="computed"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            parameter_name="delta_window",
            value=MATCH.delta_window
        ),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="sample_rate", role="shape_metadata", shape=MATCH.sample_rate),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="output_num_frames", role="shape_metadata", shape=MATCH.num_frames),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="input_mfcc_coefficients", role="shape_metadata", shape=MATCH.num_mfcc_coefficients),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="concatenated_feature_coefficients", role="shape_metadata", shape=MATCH.concatenated_coefficients),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="output_channels", role="shape_metadata", shape=MATCH.channels),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="delta_features", tensor_name="delta_feature_concatenation_adapter", role="adapter_output", shape=MATCH.adapter_shape),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="delta_features",
            weight_name="delta_feature_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_delta_features_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        internal_layers,
        combine_operation,
        delta_window,
        sample_rate,
        num_frames,
        num_mfcc_coefficients,
        concatenated_coefficients,
        channels,
        adapter_shape,
        parameter_shape
    ):
        print_delta_features_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="delta_features",
            layer_kind="speech_feature_extraction_component_group",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            internal_layers=internal_layers,
            combine_operation=combine_operation,
            sample_rate=int(sample_rate),
            num_frames=int(num_frames),
            num_mfcc_coefficients=int(num_mfcc_coefficients),
            concatenated_coefficients=int(concatenated_coefficients),
            delta_window=int(delta_window),
            adapter_shape=adapter_shape,
            channels=int(channels)
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
