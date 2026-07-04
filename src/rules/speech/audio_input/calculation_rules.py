from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    WeightShape,
    Layer,
    LayerMetadata,
    CalculationStep,
)


class AudioInputCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="audio_input",
            layer_title=MATCH.layer_title
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
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_audio_input_shape(
        self,
        layer_index,
        layer_title,
        batch_size,
        duration_seconds,
        sample_rate,
        channels,
        bit_depth
    ):
        sample_count = duration_seconds * sample_rate
        input_shape = (
            f"audio_metadata(batch_size={batch_size}, duration_seconds={duration_seconds}, "
            f"sample_rate={sample_rate}, channels={channels}, bit_depth={bit_depth})"
        )
        output_shape = f"({batch_size}, {sample_count}, {channels})"
        parameter_shape = "None (audio input has no learnable weights)"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="audio_input",
            weight_name="audio_input_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="audio_input",
            tensor_name="audio_waveform",
            role="output",
            dims_count=3,
            shape=output_shape,
            data_type="float_audio_waveform_tensor",
            source="decoded_audio_samples_with_channel_dimension_preserved"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="audio_input",
            tensor_name="samples_per_channel",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_count),
            data_type="integer_dimension_value",
            source="duration_seconds_times_sample_rate"
        ))
        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="audio_input",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source="batch_size_parameter"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="audio_input",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate),
            data_type="integer_dimension_value",
            source="sample_rate_parameter"
        ))


        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="audio_input",
            tensor_name="channel_count",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels),
            data_type="integer_dimension_value",
            source="user_selected_channel_count"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="audio_input",
            layer_kind="input_layer",
            layer_role="speech_audio_preparation",
            repeat_policy="once_first_only",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="audio_input",
            step_key=f"layer_{layer_index}_audio_samples_per_channel",
            previous_step_key="none",
            step_order=1,
            title="Samples per channel",
            formula="duration_seconds × sample_rate",
            substitution=f"{duration_seconds} × {sample_rate}",
            result=str(sample_count)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="audio_input",
            step_key=f"layer_{layer_index}_audio_output_shape",
            previous_step_key=f"layer_{layer_index}_audio_samples_per_channel",
            step_order=2,
            title="Audio waveform output shape",
            formula="(batch_size, samples_per_channel, channels)",
            substitution=f"({batch_size}, {sample_count}, {channels})",
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="audio_input",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
