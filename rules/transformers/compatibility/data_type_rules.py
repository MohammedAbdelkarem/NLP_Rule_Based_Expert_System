from experta import Rule, MATCH, NOT, AS

from facts import SelectedLayer, OptionSelection, RollbackRequest


class DataTypeCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm",
            question_key="input_data_kind",
            value="raw_token_ids",
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="layer_norm"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="layer_norm"
        )),
        salience=80
    )
    def rollback_layer_norm_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="layer_norm",
            reason="raw_token_ids_need_input_encoding_first",
            message=(
                "LayerNorm cannot be applied directly to raw integer token IDs. "
                "Choose Input Encoding first, or provide an already encoded float tensor."
            )
        ))


    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            value="raw_token_ids",
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="multi_head_attention"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention"
        )),
        salience=80
    )
    def rollback_multi_head_attention_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            reason="raw_token_ids_need_input_encoding_before_attention",
            message=(
                "Multi-Head Attention cannot be applied directly to raw integer token IDs. "
                "Choose Input Encoding first, or provide an already encoded float tensor."
            )
        ))


    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            value="raw_token_ids",
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feed_forward"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward"
        )),
        salience=80
    )
    def rollback_feed_forward_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            reason="raw_token_ids_need_input_encoding_before_feed_forward",
            message=(
                "Feed-Forward Network cannot be applied directly to raw integer token IDs. "
                "Choose Input Encoding first, or provide an already encoded float tensor."
            )
        ))
