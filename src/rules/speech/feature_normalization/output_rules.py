from experta import Rule, MATCH, NOT

from src.facts import (
    Layer,
    LayerParameter,
    TensorInfo,
    WeightShape,
    OutputStatus,
)
from src.utils.print_helpers import print_feature_normalization_result


class FeatureNormalizationOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            parameter_name="normalization_mode",
            value=MATCH.normalization_mode
        ),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="output_num_frames", role="shape_metadata", shape=MATCH.num_frames),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="output_feature_coefficients", role="shape_metadata", shape=MATCH.num_features),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="concatenated_feature_coefficients", role="shape_metadata", shape=MATCH.concatenated_features),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="output_channels", role="shape_metadata", shape=MATCH.channels),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="normalization_mode", role="shape_metadata", shape=MATCH.normalization_mode_name),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="statistics_shape", role="shape_metadata", shape=MATCH.statistics_shape),
        TensorInfo(layer_index=MATCH.layer_index, layer_name="feature_normalization", tensor_name="normalized_delta_feature_concatenation_adapter", role="adapter_suggestion", shape=MATCH.adapter_shape),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization",
            weight_name="feature_normalization_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_feature_normalization_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        normalization_mode,
        normalization_mode_name,
        num_frames,
        num_features,
        concatenated_features,
        channels,
        statistics_shape,
        adapter_shape,
        parameter_shape
    ):
        print_feature_normalization_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="feature_normalization",
            layer_kind="speech_feature_normalization_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            normalization_mode_choice=int(normalization_mode),
            normalization_mode_name=normalization_mode_name,
            num_frames=int(num_frames),
            num_features=int(num_features),
            concatenated_features=int(concatenated_features),
            channels=int(channels),
            statistics_shape=statistics_shape,
            adapter_shape=adapter_shape,
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
