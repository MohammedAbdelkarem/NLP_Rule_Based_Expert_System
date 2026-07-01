from experta import Rule, MATCH, NOT, TEST

from src.facts import (
    SelectedLayer,
    TensorInfo,
    RollbackRequest,
    TransformationSuggestion
)


class TensorAcceptanceCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="multi_head_attention"
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
            layer_name="multi_head_attention"
        )),
        salience=80
    )
    def rollback_multi_head_attention_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            from_shape=previous_shape,
            to_shape="float_hidden_tensor with shape (batch_size, sequence_length, model_dim)",
            method="Use Input Encoding or another suitable adapter before Multi-Head Attention."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            reason="multi_head_attention_requires_float_hidden_tensor",
            message=(
                "Multi-Head Attention requires a float hidden tensor. "
                "The previous layer output data type is not compatible."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="multi_head_attention"
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
            layer_name="multi_head_attention"
        )),
        salience=80
    )
    def rollback_multi_head_attention_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, model_dim)",
            method="Use a suitable adapter to produce a rank-3 hidden tensor before Multi-Head Attention."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            reason="multi_head_attention_requires_rank_3_tensor",
            message=(
                "Multi-Head Attention requires a rank-3 hidden tensor with shape "
                "(batch_size, sequence_length, model_dim)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
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
            layer_name="residual_connection"
        )),
        salience=80
    )
    def rollback_residual_connection_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape=previous_shape,
            to_shape="float_hidden_tensor with shape (batch_size, sequence_length, model_dim)",
            method="Use a suitable adapter before Residual Connection or choose another layer."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="residual_connection",
            reason="residual_connection_requires_float_hidden_tensor",
            message=(
                "Residual Connection requires a float hidden tensor as the previous output. "
                "The previous layer output data type is not compatible."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
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
            layer_name="residual_connection"
        )),
        salience=80
    )
    def rollback_residual_connection_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, model_dim)",
            method="Use a suitable adapter to produce a rank-3 hidden tensor before Residual Connection."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="residual_connection",
            reason="residual_connection_requires_rank_3_tensor",
            message=(
                "Residual Connection requires a rank-3 hidden tensor with shape "
                "(batch_size, sequence_length, model_dim)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feed_forward"
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
            layer_name="feed_forward"
        )),
        salience=80
    )
    def rollback_feed_forward_for_invalid_previous_data_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feed_forward",
            from_shape=previous_shape,
            to_shape="float_hidden_tensor with shape (batch_size, sequence_length, model_dim)",
            method="Use Input Encoding or another suitable adapter before Feed-Forward Network."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            reason="feed_forward_requires_float_hidden_tensor",
            message=(
                "Feed-Forward Network requires a float hidden tensor. "
                "The previous layer output data type is not compatible."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feed_forward"
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
            layer_name="feed_forward"
        )),
        salience=80
    )
    def rollback_feed_forward_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="feed_forward",
            from_shape=previous_shape,
            to_shape="(batch_size, sequence_length, model_dim)",
            method="Use a suitable adapter to produce a rank-3 hidden tensor before Feed-Forward Network."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            reason="feed_forward_requires_rank_3_tensor",
            message=(
                "Feed-Forward Network requires a rank-3 hidden tensor with shape "
                "(batch_size, sequence_length, model_dim)."
            )
        ))
