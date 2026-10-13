from experta import AS, MATCH, NOT, Rule, TEST

from facts import (
    OptionSelection,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


class SigmoidActivationCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            value=MATCH.input_data_type,
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="sigmoid",
        ),
        TEST(lambda input_data_type: input_data_type != "binary_logits"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="sigmoid")),
        salience=80,
    )
    def rollback_first_sigmoid_for_non_logits(
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
            layer_name="sigmoid",
            reason="sigmoid_requires_binary_logits",
            message=(
                "Sigmoid output activation requires binary logits with last dimension = "
                "1. Use a binary linear projection or Binary Logistic Regression Head."
            ),
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="sigmoid",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        TEST(lambda data_type: data_type != "binary_logits"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="sigmoid")),
        salience=80,
    )
    def rollback_sigmoid_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(
            TransformationSuggestion(
                layer_index=layer_index,
                layer_name="sigmoid",
                from_shape=previous_shape,
                to_shape="binary logits with last dimension = 1",
                method=(
                    "Use Binary Logistic Regression Head, or a linear projection with "
                    "output_dim = 1 and binary logits purpose."
                ),
            ),
        )
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="sigmoid",
            reason="sigmoid_requires_binary_logits",
            message=(
                "Sigmoid requires binary logits as input. The previous layer output data "
                "type is not compatible."
            ),
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="sigmoid",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="binary_logits",
        ),
        TEST(lambda dims_count: dims_count != 2 and dims_count != 3),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="sigmoid")),
        salience=80,
    )
    def rollback_sigmoid_for_invalid_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="sigmoid",
            reason="sigmoid_requires_rank_2_or_rank_3_logits",
            message="Sigmoid accepts rank-2 logits (B, 1) or rank-3 logits (B, T, 1).",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="sigmoid",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="binary_logits",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            shape=MATCH.output_dim,
        ),
        TEST(lambda output_dim: str(output_dim) != "1"),
        NOT(RollbackRequest(layer_index=MATCH.layer_index, layer_name="sigmoid")),
        salience=80,
    )
    def rollback_sigmoid_for_non_binary_logit_width(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        output_dim
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="sigmoid",
            reason="sigmoid_requires_single_logit",
            message=(
                "Sigmoid binary output activation requires the last dimension to be 1. "
                "Use Softmax for multi-class logits."
            ),
        ))
