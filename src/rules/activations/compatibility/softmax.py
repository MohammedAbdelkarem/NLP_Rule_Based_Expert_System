from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import (
    SelectedLayer,
    OptionSelection,
    TensorInfo,
    RollbackRequest,
    TransformationSuggestion
)


class SoftmaxActivationCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="softmax",
            question_key="softmax_input_data_kind",
            value=MATCH.input_data_type,
            status="resolved"
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="softmax"
        ),
        TEST(lambda input_data_type: input_data_type != "logits"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="softmax"
        )),
        salience=80
    )
    def rollback_first_softmax_for_non_logits(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
        input_data_type
    ):
        self.retract(option_fact)

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            reason="softmax_requires_logits",
            message=(
                "Softmax requires logits as input. "
                "Use Position-wise Linear with output score projection before Softmax."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="softmax"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "logits"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="softmax"
        )),
        salience=80
    )
    def rollback_softmax_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="softmax",
            from_shape=previous_shape,
            to_shape="logits with shape (batch_size, sequence_length, output_dim)",
            method="Use Position-wise Linear with output score projection before Softmax."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            reason="softmax_requires_logits",
            message=(
                "Softmax requires logits as input. "
                "The previous layer output data type is not compatible."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="softmax"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="logits"
        ),
        TEST(lambda dims_count: dims_count != 3),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="softmax"
        )),
        salience=80
    )
    def rollback_softmax_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="softmax",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, output_dim)",
            method="Use an adapter to produce rank-3 logits before Softmax."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            reason="softmax_requires_rank_3_logits",
            message=(
                "Softmax requires rank-3 logits with shape "
                "(batch_size, sequence_length, output_dim)."
            )
        ))
