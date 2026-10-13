from experta import Rule, MATCH, NOT, TEST

from facts import SelectedLayer, RollbackRequest, TransformationSuggestion


class InputOrderCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="input_encoding"
        ),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding"
        )),
        salience=82
    )
    def rollback_input_encoding_after_previous_layer(
        self,
        layer_index,
        previous_layer_index
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="input_encoding",
            reason="input_encoding_must_be_first",
            message=(
                "Input Encoding must be the first layer because it works on raw token IDs. "
                "Choose another layer that accepts the previous output shape."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="residual_connection"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection"
        )),
        salience=82
    )
    def rollback_residual_connection_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape="no previous tensor",
            to_shape="two compatible hidden tensors with the same shape",
            method="Add at least one hidden-tensor-producing layer before Residual Connection."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="residual_connection",
            reason="residual_connection_needs_previous_tensor",
            message=(
                "Residual Connection cannot be the first layer. "
                "It needs a previous layer output and an earlier compatible source tensor."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=1,
            layer_name="residual_connection"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection"
        )),
        salience=82
    )
    def rollback_residual_connection_without_earlier_source(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape="only one previous tensor",
            to_shape="previous layer output plus an earlier compatible source tensor",
            method="Add another hidden-tensor layer before using Residual Connection."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=1,
            layer_name="residual_connection",
            reason="residual_connection_needs_earlier_source_tensor",
            message=(
                "Residual Connection needs two tensors: the previous layer output and an earlier source tensor. "
                "There is not enough previous architecture history yet."
            )
        ))
