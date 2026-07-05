from experta import AS, MATCH, NOT, Rule

from src.facts import (
    OptionSelection,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


class FeedforwardClassifierCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            value="raw_token_ids",
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feedforward_classifier_head",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
        )),
        salience=80,
    )
    def rollback_classifier_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
    ):
        self.retract(option_fact)
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_classifier_head",
            reason="raw_token_ids_need_embedding_before_classifier",
            message=(
                "A classifier head cannot operate directly on raw token IDs. "
                "Use Input Encoding or an embedding layer first."
            ),
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            value="sequence_hidden_tensor",
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feedforward_classifier_head",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
        )),
        salience=80,
    )
    def rollback_first_classifier_for_sequence_input(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
    ):
        self.retract(option_fact)
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_classifier_head",
            reason="classifier_requires_feature_vector",
            message=(
                "The feedforward classifier head expects one feature vector per example. "
                "Use the Embedding-to-Classifier Pooling Adapter for sequence tensors."
            ),
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feedforward_classifier_head",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
        )),
        salience=80,
    )
    def rollback_classifier_for_previous_sequence_tensor(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            from_shape=previous_shape,
            to_shape="(batch_size, feature_dim)",
            method="Use Embedding-to-Classifier Mean Pooling Adapter before the classifier head.",
        ))
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_classifier_head",
            reason="classifier_requires_rank_2_feature_vector",
            message="Classifier head requires rank-2 feature vectors, not a sequence tensor.",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feedforward_classifier_head",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type,
        ),
        NOT(TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            data_type="float_hidden_tensor",
        )),
        NOT(TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            data_type="float_hidden_tensor",
        )),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
        )),
        salience=79,
    )
    def rollback_classifier_for_invalid_previous_tensor(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        data_type,
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            from_shape=previous_shape,
            to_shape="(batch_size, feature_dim)",
            method="Use an embedding, pooling, or projection adapter before classification.",
        ))
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feedforward_classifier_head",
            reason="classifier_requires_float_feature_vector",
            message="Classifier head requires a rank-2 float hidden tensor.",
        ))
