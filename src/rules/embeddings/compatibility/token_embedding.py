from experta import Rule, MATCH, NOT, TEST

from src.facts import SelectedLayer, RollbackRequest


class TokenEmbeddingCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="token_embedding"
        ),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="token_embedding"
        )),
        salience=82
    )
    def rollback_token_embedding_after_previous_layer(
        self,
        layer_index,
        previous_layer_index
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="token_embedding",
            reason="token_embedding_must_be_first",
            message=(
                "Token Embedding must be the first layer because it works on raw "
                "integer token IDs. Choose a layer that accepts the previous "
                "float tensor output instead."
            )
        ))
