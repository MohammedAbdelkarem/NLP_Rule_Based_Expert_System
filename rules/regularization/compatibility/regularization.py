from experta import MATCH, NOT, Rule, TEST

from facts import LayerMetadata, RollbackRequest, SelectedLayer, WeightShape




def is_non_trainable_weight_shape(previous_weights):
    value = str(previous_weights).strip().lower()
    return value == "no trainable parameters" or "no trainable" in value


def is_training_component_role(previous_role):
    return previous_role in (
        "loss_function",
        "regularization_constraint",
    )


class RegularizationCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="l1_l2_regularization",
        ),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="l1_l2_regularization")),
        salience=82
    )
    def rollback_regularization_as_first_component(self, layer_index):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=0,
                layer_name="l1_l2_regularization",
                reason="regularization_cannot_be_first",
                message=(
                    "Regularization is a training constraint. It must be attached after a "
                    "trainable layer or classification head."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="l1_l2_regularization",
        ),
        WeightShape(layer_index=MATCH.previous_layer_index, shape=MATCH.previous_weights),
        TEST(lambda previous_weights: is_non_trainable_weight_shape(previous_weights)),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="l1_l2_regularization")),
        salience=81
    )
    def rollback_regularization_for_non_trainable_previous(
        self,
        layer_index,
        previous_layer_index
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="l1_l2_regularization",
                reason="regularization_requires_trainable_weights",
                message=(
                    "L1/L2 regularization must be applied to trainable weights. The previous "
                    "component has no trainable parameters."
                ),
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="l1_l2_regularization",
        ),
        LayerMetadata(layer_index=MATCH.previous_layer_index, layer_role=MATCH.previous_role),
        TEST(lambda previous_role: is_training_component_role(previous_role)),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="l1_l2_regularization")),
        salience=81
    )
    def rollback_regularization_after_training_component(
        self,
        layer_index,
        previous_layer_index,
        previous_role
    ):
        self.declare(
            RollbackRequest(
                layer_index=layer_index,
                previous_layer_index=previous_layer_index,
                layer_name="l1_l2_regularization",
                reason="regularization_position_invalid",
                message=(
                    "Regularization should be attached to trainable model parameters before "
                    "the loss component, not after another training component."
                ),
            ),
        )
