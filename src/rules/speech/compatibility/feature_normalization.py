from experta import Rule, MATCH, NOT, TEST

from src.facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
)


class FeatureNormalizationCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="feature_normalization"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization"
        )),
        salience=83
    )
    def rollback_feature_normalization_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feature_normalization",
            from_shape="no previous Delta feature stream group",
            to_shape="mfcc=(B, frames, F, C); delta=(B, frames, F, C); delta_delta=(B, frames, F, C)",
            method="Add MFCC and Delta + Delta-Delta feature augmentation before normalization."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="feature_normalization",
            reason="feature_normalization_needs_delta_streams",
            message=(
                "Speech Feature Normalization cannot be the first layer. "
                "It needs separated MFCC, Delta, and Delta-Delta streams."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feature_normalization"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "separate_delta_feature_stream_group"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization"
        )),
        salience=82
    )
    def rollback_feature_normalization_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feature_normalization",
            from_shape=previous_shape,
            to_shape="separated MFCC + Delta + Delta-Delta feature stream group",
            method="Use Delta + Delta-Delta Feature Augmentation before Speech Feature Normalization."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feature_normalization",
            reason="feature_normalization_requires_separated_delta_streams",
            message=(
                "Speech Feature Normalization requires the separated output from "
                "Delta + Delta-Delta Feature Augmentation."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feature_normalization"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="separate_delta_feature_stream_group"
        ),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feature_normalization"
        )),
        salience=82
    )
    def rollback_feature_normalization_for_invalid_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feature_normalization",
            from_shape=previous_shape,
            to_shape="three rank-4 streams: (B, frames, features, channels)",
            method="Normalize separated rank-4 Delta feature streams."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feature_normalization",
            reason="feature_normalization_requires_rank_4_stream_group",
            message=(
                "Speech Feature Normalization expects a separated stream group whose streams are rank-4: "
                "(batch_size, num_frames, feature_coefficients, channels)."
            )
        ))
