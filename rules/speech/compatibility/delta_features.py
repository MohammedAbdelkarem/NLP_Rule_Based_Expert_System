from experta import Rule, MATCH, NOT, TEST

from facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
)


class DeltaFeatureCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="delta_features"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="delta_features"
        )),
        salience=83
    )
    def rollback_delta_features_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="delta_features",
            from_shape="no previous MFCC tensor",
            to_shape="(batch_size, num_frames, num_mfcc_coefficients, channels)",
            method="Add the speech feature pipeline through MFCC before Delta + Delta-Delta features."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="delta_features",
            reason="delta_features_need_mfcc_output",
            message=(
                "Delta + Delta-Delta features cannot be the first layer. "
                "They need MFCC output with shape "
                "(batch_size, num_frames, num_mfcc_coefficients, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="delta_features"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "mfcc_feature_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="delta_features"
        )),
        salience=82
    )
    def rollback_delta_features_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="delta_features",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mfcc_coefficients, channels)",
            method="Use MFCC Component Group output before Delta + Delta-Delta features."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="delta_features",
            reason="delta_features_require_mfcc_tensor",
            message=(
                "Delta + Delta-Delta features require an MFCC tensor. "
                "Add MFCC before this component group."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="delta_features"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="mfcc_feature_tensor"
        ),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="delta_features"
        )),
        salience=82
    )
    def rollback_delta_features_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="delta_features",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mfcc_coefficients, channels)",
            method="Use rank-4 MFCC features before Delta + Delta-Delta features."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="delta_features",
            reason="delta_features_require_rank_4_mfcc_tensor",
            message=(
                "Delta + Delta-Delta features require a rank-4 MFCC tensor with shape "
                "(batch_size, num_frames, num_mfcc_coefficients, channels)."
            )
        ))
