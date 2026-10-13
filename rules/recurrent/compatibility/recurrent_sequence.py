from experta import MATCH, NOT, Rule, TEST

from facts import Layer, RollbackRequest, SelectedLayer, TensorInfo


class RecurrentInputCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name
        ),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        TEST(lambda layer_name: layer_name in (
            "simple_rnn",
            "lstm",
            "stacked_rnn",
            "bidirectional_rnn",
            "rnn_encoder_decoder",
        )),
        Layer(layer_index=MATCH.previous_layer_index),
        NOT(TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length"
        )),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name
        )),
        salience=83
    )
    def rollback_recurrent_after_non_sequence_output(
        self,
        layer_index,
        previous_layer_index,
        layer_name
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name=layer_name,
            reason="recurrent_component_requires_sequence_tensor",
            message=(
                "This Chapter 13 recurrent component requires a 3D sequence "
                "tensor shaped as (batch_size, sequence_length, feature_dim)."
            )
        ))
