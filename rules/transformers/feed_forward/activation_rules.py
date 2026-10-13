from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    OptionSelection,
    IntermediateTensor,
    CalculationStep
)


class FeedForwardActivationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            value="relu",
            status="resolved"
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="first_projection_output",
            shape=MATCH.first_projection_shape
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="activation_output"
        )),
        salience=63
    )
    def calculate_feed_forward_relu_activation_shape(
        self,
        layer_index,
        first_projection_shape
    ):
        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="activation_output",
            shape=first_projection_shape,
            data_type="float_hidden_tensor",
            source="relu_activation_after_first_projection"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="feed_forward",
            step_key=f"layer_{layer_index}_ffn_activation_shape",
            previous_step_key=f"layer_{layer_index}_ffn_first_projection_shape",
            step_order=2,
            title="Feed-Forward ReLU activation shape",
            formula="ReLU preserves tensor shape",
            substitution=f"ReLU input shape = {first_projection_shape}",
            result=first_projection_shape
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            value="gelu",
            status="resolved"
        ),
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="first_projection_output",
            shape=MATCH.first_projection_shape
        ),
        NOT(IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            tensor_name="activation_output"
        )),
        salience=63
    )
    def calculate_feed_forward_gelu_activation_shape(
        self,
        layer_index,
        first_projection_shape
    ):
        self.declare(IntermediateTensor(
            layer_index=layer_index,
            layer_name="feed_forward",
            tensor_name="activation_output",
            shape=first_projection_shape,
            data_type="float_hidden_tensor",
            source="gelu_activation_after_first_projection"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="feed_forward",
            step_key=f"layer_{layer_index}_ffn_activation_shape",
            previous_step_key=f"layer_{layer_index}_ffn_first_projection_shape",
            step_order=2,
            title="Feed-Forward GELU activation shape",
            formula="GELU preserves tensor shape",
            substitution=f"GELU input shape = {first_projection_shape}",
            result=first_projection_shape
        ))
