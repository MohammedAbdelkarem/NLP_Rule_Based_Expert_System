from experta import MATCH, NOT, Rule, TEST

from facts import (
    LayerMetadata,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


def is_classification_head(layer_name):
    return layer_name in(
        "binary_logistic_regression_head",
        "multinomial_logistic_regression_head",
    )


class ClassificationHeadCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: is_classification_head(layer_name)),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(lambda data_type: data_type != "float_hidden_tensor"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name=MATCH.layer_name)),
        salience=80,
    )
    def rollback_classification_head_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        layer_name,
        previous_shape,
        data_type
    ):
        self.declare(
            TransformationSuggestion(
                layer_index=layer_index,
                layer_name=layer_name,
                from_shape=previous_shape,
                to_shape="float hidden tensor with shape (B, input_dim) or (B, T, input_dim)",
                method=(
                    "Use an embedding/encoder/projection layer that outputs float hidden "
                    "features before the classification head."
                ),
            ),
        )
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name=layer_name,
                reason="classification_head_requires_float_hidden_tensor",
                message=(
                    "Classification heads require float hidden features, not logits, "
                    "probabilities, raw tokens, or speech preprocessing tensors."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: is_classification_head(layer_name)),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor",
        ),
        TEST(lambda dims_count: dims_count != 2 and dims_count != 3),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name=MATCH.layer_name)),
        salience=80,
    )
    def rollback_classification_head_for_invalid_rank(
        self,
        layer_index,
        previous_layer_index,
        layer_name,
        previous_shape
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name=layer_name,
                reason="classification_head_requires_rank_2_or_rank_3",
                message=(
                    "Classification heads accept rank-2 vectors (B, input_dim) or rank-3 "
                    "sequence features (B, T, input_dim)."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: is_classification_head(layer_name)),
        LayerMetadata(layer_index=MATCH.previous_layer_index, layer_role="classification_head"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name=MATCH.layer_name)),
        salience=81,
    )
    def rollback_consecutive_classification_heads(
        self,
        layer_index,
        previous_layer_index,
        layer_name
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name=layer_name,
                reason="consecutive_classification_heads",
                message=(
                    "A classification head is normally terminal. Do not place another "
                    "classification head directly after probabilities."
                ),
            ),
        )
