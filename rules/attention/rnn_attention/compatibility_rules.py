from experta import MATCH, NOT, Rule, TEST

from facts import OptionSelection, RollbackRequest, SelectedLayer


class RNNAttentionCompatibilityRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=MATCH.previous_layer_index, layer_name="rnn_attention"),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="rnn_attention")),
        salience=83
    )
    def rollback_rnn_attention_after_previous_layer(self, layer_index, previous_layer_index):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="rnn_attention",
            reason="rnn_attention_is_standalone_shape_simulation",
            message="RNN Attention is simulated as a standalone Chapter 13 component."
        ))

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_attention"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", value="dot_product", status="resolved"),
        OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_hidden_match", value="mismatch", status="resolved"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="rnn_attention")),
        salience=82
    )
    def rollback_dot_attention_hidden_mismatch(self, layer_index):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="rnn_attention",
            reason="dot_product_attention_requires_matching_hidden_sizes",
            message="Dot-product attention requires encoder and decoder hidden sizes to match."
        ))
