from experta import Rule, MATCH, NOT, TEST

from src.facts import SelectedLayer, Layer, LayerMetadata, RollbackRequest


class ConsecutiveLayerCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.current_layer_name
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.current_layer_name
        ),
        LayerMetadata(
            layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.current_layer_name,
            consecutive_repeat_policy="no_consecutive_repeat"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.current_layer_name
        )),
        salience=81
    )
    def rollback_consecutive_duplicate_layer(
        self,
        layer_index,
        previous_layer_index,
        current_layer_name
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=current_layer_name,
            reason="consecutive_duplicate_layer",
            message=(
                "This layer was selected directly after the same layer. "
                "Consecutive duplicate layers are usually redundant. "
                "Choose a different layer for this position."
            )
        ))
