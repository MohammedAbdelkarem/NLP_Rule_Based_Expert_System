from experta import Rule, MATCH, NOT, TEST

from src.facts import SelectedLayer, RollbackRequest


class AudioInputCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="audio_input"
        ),
        TEST(lambda previous_layer_index: previous_layer_index != 0),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="audio_input"
        )),
        salience=82
    )
    def rollback_audio_input_after_previous_layer(
        self,
        layer_index,
        previous_layer_index
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="audio_input",
            reason="audio_input_must_be_first",
            message=(
                "Audio Input must be the first layer because it works on raw audio/WAV metadata. "
                "Choose another layer that accepts the previous output shape."
            )
        ))
