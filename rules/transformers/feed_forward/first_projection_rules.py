from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    IntermediateTensor,
    CalculationStep
)


class FeedForwardFirstProjectionRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="sequence_length",
            value=MATCH.sequence_length
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=MATCH.feed_forward_dim
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="first_projection_output"
        )),
        salience=64
    )
    def calculate_feed_forward_first_projection_shape(
        self,
        layer_index,
        batch_size,
        sequence_length,
        model_dim,
        feed_forward_dim
    ):
        first_projection_shape = f"({batch_size}, {sequence_length}, {feed_forward_dim})"

        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="first_projection_output",
            shape=first_projection_shape,
            data_type="float_hidden_tensor",
            source="position_wise_linear_projection_W1_b1"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="feed_forward",
            step_key=f"layer_{layer_index}_ffn_first_projection_shape",
            previous_step_key="none",
            step_order=1,
            title="Feed-Forward first projection shape",
            formula="(batch_size, sequence_length, model_dim) × W1(model_dim, d_ff)",
            substitution=f"({batch_size}, {sequence_length}, {model_dim}) × ({model_dim}, {feed_forward_dim})",
            result=first_projection_shape
        ))
