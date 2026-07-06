from experta import Rule, MATCH, NOT

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    AUDIO_INPUT_SELECTED,
    ASK_AUDIO_BATCH_SIZE,
    ASK_AUDIO_DURATION_SECONDS,
    ASK_AUDIO_SAMPLE_RATE,
    ASK_AUDIO_CHANNELS,
    ASK_AUDIO_BIT_DEPTH,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class AudioInputParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="audio_input"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_audio_input_parameters(self, layer_index):
        print(AUDIO_INPUT_SELECTED)

        batch_size = read_int_parameter(ASK_AUDIO_BATCH_SIZE)
        duration_seconds = read_int_parameter(ASK_AUDIO_DURATION_SECONDS)
        sample_rate = read_int_parameter(ASK_AUDIO_SAMPLE_RATE)
        channels = read_int_parameter(ASK_AUDIO_CHANNELS)
        bit_depth = read_int_parameter(ASK_AUDIO_BIT_DEPTH)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="audio_input",
            parameter_name="batch_size",
            value=batch_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="audio_input",
            parameter_name="duration_seconds",
            value=duration_seconds
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=sample_rate
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="audio_input",
            parameter_name="channels",
            value=channels
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="audio_input",
            parameter_name="bit_depth",
            value=bit_depth
        ))
