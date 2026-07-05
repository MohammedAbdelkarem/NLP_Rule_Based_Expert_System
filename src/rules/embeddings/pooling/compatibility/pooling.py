from experta import AS, MATCH, NOT, Rule

from src.facts import (
    OptionSelection,
    RollbackRequest,
    SelectedLayer,
    TensorInfo,
    TransformationSuggestion,
)


class EmbeddingPoolingCompatibilityRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            value="raw_token_ids",
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
        )),
        salience=80,
    )
    def rollback_pooling_for_raw_tokens(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
    ):
        self.retract(option_fact)
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
            reason="raw_token_ids_need_embedding_before_pooling",
            message="Pooling requires a sequence of float embeddings, not raw token IDs.",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            value="feature_vector_tensor",
            status="resolved",
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
        )),
        salience=80,
    )
    def rollback_first_pooling_for_feature_vector(
        self,
        option_fact,
        layer_index,
        previous_layer_index,
    ):
        self.retract(option_fact)
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
            reason="pooling_requires_sequence_tensor",
            message="Pooling is only needed for rank-3 sequence tensors shaped (B, T, d).",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            shape=MATCH.previous_shape,
            data_type="float_hidden_tensor",
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
        )),
        salience=80,
    )
    def rollback_pooling_for_previous_feature_vector(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            from_shape=previous_shape,
            to_shape="already compatible with classifier head",
            method="Skip pooling and connect directly to the classifier head.",
        ))
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
            reason="pooling_not_needed_for_rank_2_tensor",
            message="The previous tensor is already rank-2 and does not need sequence pooling.",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
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
            dims_count=3,
            data_type="float_hidden_tensor",
        )),
        NOT(TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            data_type="float_hidden_tensor",
        )),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
        )),
        salience=79,
    )
    def rollback_pooling_for_invalid_previous_tensor(
        self,
        layer_index,
        previous_layer_index,
        previous_shape,
        data_type,
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, embedding_dim)",
            method="Use Input Encoding or another embedding layer before pooling.",
        ))
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="embedding_to_classifier_pooling",
            reason="pooling_requires_rank_3_embedding_tensor",
            message="Pooling requires a rank-3 float hidden tensor.",
        ))
