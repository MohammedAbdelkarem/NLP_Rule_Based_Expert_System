from experta import Rule, MATCH, NOT, AS, TEST

from facts import (
    SelectedLayer,
    OptionSelection,
    TensorInfo,
    RollbackRequest,
    TransformationSuggestion
)


class LinearProjectionCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            value="raw_token_ids",
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="position_wise_linear"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear"
        )),
        salience=80
    )
    def rollback_position_wise_linear_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            reason="raw_token_ids_need_input_encoding_before_linear",
            message=(
                "Position-wise Linear cannot be applied directly to raw integer token IDs. "
                "Choose Input Encoding first, or provide an already encoded float tensor."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="position_wise_linear"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "float_hidden_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear"
        )),
        salience=80
    )
    def rollback_position_wise_linear_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            from_shape=previous_shape,
            to_shape="float_hidden_tensor with shape (batch_size, sequence_length, feature_dim)",
            method="Use Input Encoding or another suitable adapter before Position-wise Linear."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            reason="position_wise_linear_requires_float_hidden_tensor",
            message=(
                "Position-wise Linear requires a float hidden tensor. "
                "The previous layer output data type is not compatible."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="position_wise_linear"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor"
        ),
        TEST(lambda dims_count: dims_count != 3),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear"
        )),
        salience=80
    )
    def rollback_position_wise_linear_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, feature_dim)",
            method="Use a suitable adapter to produce a rank-3 hidden tensor before Position-wise Linear."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            reason="position_wise_linear_requires_rank_3_tensor",
            message=(
                "Position-wise Linear requires a rank-3 hidden tensor with shape "
                "(batch_size, sequence_length, feature_dim)."
            )
        ))
