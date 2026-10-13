from experta import MATCH, NOT, Rule, TEST

from facts import LayerParameter, RollbackRequest, SelectedLayer


class PretrainedStaticEmbeddingCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="pretrained_static_embedding",
        ),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        NOT(
            RollbackRequest(
                layer_index=MATCH.layer_index,
                layer_name="pretrained_static_embedding",
            ),
        ),
        salience=82,
    )
    def rollback_pretrained_static_embedding_after_previous_layer(
        self,
        layer_index,
        previous_layer_index
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="pretrained_static_embedding",
            reason="pretrained_static_embedding_must_be_first",
            message=(
                "Pretrained Static Embedding must be the first layer because it "
                "works on raw integer token IDs. Choose a layer that accepts the "
                "previous float tensor output instead."
            ),
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name=MATCH.parameter_name,
            value=MATCH.value,
        ),
        TEST(lambda value: value <= 0),
        NOT(
            RollbackRequest(
                layer_index=MATCH.layer_index,
                layer_name="pretrained_static_embedding",
            ),
        ),
        salience=81,
    )
    def rollback_pretrained_static_embedding_for_invalid_dimension(
        self,
        layer_index,
        parameter_name,
        value
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
            reason="pretrained_static_embedding_dimensions_must_be_positive",
            message=(
                "Pretrained Static Embedding requires positive integers for "
                "batch size, sequence length, vocabulary size, and embedding dim."
            ),
        ))
