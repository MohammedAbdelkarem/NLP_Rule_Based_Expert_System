from experta import AS, MATCH, NOT, Rule, TEST

from src.facts import (
    OptionSelection,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


class FeedforwardCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="feedforward_input_kind",
            value="raw_token_ids",
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: layer_name == "hidden_dense_activation_block" or layer_name == "feedforward_mlp_block"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
        )),
        salience=80,
    )
    def rollback_feedforward_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
        layer_name,
    ):
        self.retract(option_fact)
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=layer_name,
            reason="raw_token_ids_need_embedding_before_feedforward",
            message=(
                "Feedforward blocks cannot operate directly on raw token IDs. "
                "Use Input Encoding or an embedding layer first."
            ),
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
        ),
        TEST(lambda layer_name: layer_name == "hidden_dense_activation_block" or layer_name == "feedforward_mlp_block"),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(lambda data_type: data_type != "float_hidden_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
        )),
        salience=80,
    )
    def rollback_feedforward_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        layer_name,
        previous_shape,
        data_type,
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name=layer_name,
            from_shape=previous_shape,
            to_shape="float hidden tensor",
            method="Use Input Encoding, embedding, pooling, or a projection adapter first.",
        ))
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=layer_name,
            reason="feedforward_requires_float_hidden_tensor",
            message="Feedforward blocks require a float hidden tensor as input.",
        ))
