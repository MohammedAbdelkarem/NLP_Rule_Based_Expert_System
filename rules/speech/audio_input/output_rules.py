from experta import Rule, MATCH, NOT

from facts import Layer, LayerParameter, WeightShape, OutputStatus
from utils.print_helpers import print_audio_input_result


class AudioInputOutputRules:
    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="duration_seconds",
            value=MATCH.duration_seconds
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=MATCH.sample_rate
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="channels",
            value=MATCH.channels
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            parameter_name="bit_depth",
            value=MATCH.bit_depth
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name="audio_input",
            weight_name="audio_input_parameters",
            shape=MATCH.parameter_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_result_printed",
            status="printed"
        )),
        salience=50
    )
    def print_audio_input_layer_result(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        batch_size,
        duration_seconds,
        sample_rate,
        channels,
        bit_depth,
        parameter_shape
    ):
        sample_count = duration_seconds * sample_rate

        print_audio_input_result(
            layer_index=layer_index,
            layer_title=layer_title,
            layer_name="audio_input",
            layer_kind="input_layer",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_shape=parameter_shape,
            batch_size=batch_size,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            sample_count=sample_count
        )

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_result_printed",
            status="printed"
        ))
