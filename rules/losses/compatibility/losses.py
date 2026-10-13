from experta import MATCH, NOT, Rule, TEST

from facts import (
    LayerMetadata,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


def is_loss(layer_name):
    return layer_name in ("binary_cross_entropy", "categorical_cross_entropy")


class LossCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: is_loss(layer_name)),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name=MATCH.layer_name)),
        salience=82
    )
    def rollback_loss_as_first_component(self, layer_index, layer_name):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=0,
                layer_name=layer_name,
                reason="loss_cannot_be_first",
                message=(
                    "A loss function is a training component. It must come after a "
                    "prediction head that outputs probabilities."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="binary_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(lambda data_type: data_type != "binary_probability"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="binary_cross_entropy")),
        salience=81
    )
    def rollback_bce_for_invalid_prediction_type(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        data_type
    ):
        self.declare(
            TransformationSuggestion(
                layer_index=layer_index,
                layer_name="binary_cross_entropy",
                from_shape=previous_shape,
                to_shape="binary probabilities with shape (B, 1) or (B, T, 1)",
                method=(
                    "Use Binary Logistic Regression Head or Sigmoid before Binary "
                    "Cross-Entropy."
                ),
            ),
        )
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="binary_cross_entropy",
                reason="bce_requires_binary_probabilities",
                message=(
                    "Binary Cross-Entropy requires binary probabilities from a "
                    "sigmoid/binary logistic regression head."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="binary_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="binary_probability",
        ),
        TEST(lambda dims_count: dims_count != 2 and dims_count != 3),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="binary_cross_entropy")),
        salience=81
    )
    def rollback_bce_for_invalid_rank(self, layer_index, previous_layer_index, previous_shape):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="binary_cross_entropy",
                reason="bce_requires_rank_2_or_rank_3",
                message=(
                    "Binary Cross-Entropy accepts predictions with shape (B, 1) or (B, T, 1)."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="binary_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type="binary_probability",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.output_dim,
        ),
        TEST(lambda output_dim: str(output_dim) != "1"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="binary_cross_entropy")),
        salience=81
    )
    def rollback_bce_for_non_binary_width(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        output_dim
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="binary_cross_entropy",
                reason="bce_requires_one_binary_probability",
                message=(
                    "Binary Cross-Entropy requires exactly one binary probability per item. "
                    "Use Categorical Cross-Entropy for multi-class distributions."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(
            lambda data_type: data_type != "class_probabilities" and data_type != "probabilities",
        ),
        NOT(
            RollbackRequest(
                layer_index=MATCH.layer_index,
                layer_name="categorical_cross_entropy",
            ),
        ),
        salience=81
    )
    def rollback_cce_for_invalid_prediction_type(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        data_type
    ):
        self.declare(
            TransformationSuggestion(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                from_shape=previous_shape,
                to_shape="class probabilities with shape (B, C) or (B, T, C)",
                method=(
                    "Use Multinomial Logistic Regression Head or Softmax before Categorical "
                    "Cross-Entropy."
                ),
            ),
        )
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="categorical_cross_entropy",
                reason="cce_requires_class_probabilities",
                message=(
                    "Categorical Cross-Entropy requires a class probability distribution "
                    "from Softmax or a multinomial logistic regression head."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(
            lambda data_type: (
                data_type == "class_probabilities"
                or data_type == "probabilities"
            ),
        ),
        TEST(lambda dims_count: dims_count != 2 and dims_count != 3),
        NOT(
            RollbackRequest(
                layer_index=MATCH.layer_index,
                layer_name="categorical_cross_entropy",
            ),
        ),
        salience=81
    )
    def rollback_cce_for_invalid_rank(self, layer_index, previous_layer_index, previous_shape):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="categorical_cross_entropy",
                reason="cce_requires_rank_2_or_rank_3",
                message=(
                    "Categorical Cross-Entropy accepts predictions with shape (B, C) or (B, "
                    "T, C)."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="categorical_cross_entropy",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(
            lambda data_type: (
                data_type == "class_probabilities"
                or data_type == "probabilities"
            ),
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.output_dim,
        ),
        TEST(lambda output_dim: int(output_dim) <= 1),
        NOT(
            RollbackRequest(
                layer_index=MATCH.layer_index,
                layer_name="categorical_cross_entropy",
            ),
        ),
        salience=81
    )
    def rollback_cce_for_single_class_width(
        self,
        layer_index,
        previous_layer_index,
        output_dim
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="categorical_cross_entropy",
                reason="cce_requires_more_than_one_class",
                message=(
                    "Categorical Cross-Entropy requires more than one class. Use Binary "
                    "Cross-Entropy for a single sigmoid probability."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: is_loss(layer_name)),
        LayerMetadata(layer_index=MATCH.previous_layer_index, layer_role="loss_function"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name=MATCH.layer_name)),
        salience=82
    )
    def rollback_consecutive_losses(self, layer_index, previous_layer_index, layer_name):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name=layer_name,
                reason="consecutive_loss_functions",
                message=(
                    "A loss function is terminal in this expert-system flow. Do not add "
                    "another loss directly after it."
                ),
            ),
        )
