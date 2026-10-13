from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    WeightShape,
    Layer,
    LayerMetadata,
    CalculationStep,
)


class FramingWindowingCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="samples_per_channel",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.samples_per_channel
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="channel_count",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.channels
        ),
        LayerParameter(
            layer_index=MATCH.previous_layer_index,
            layer_name="audio_input",
            parameter_name="batch_size",
            value=MATCH.batch_size
        ),
        LayerParameter(
            layer_index=MATCH.previous_layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=MATCH.sample_rate
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            parameter_name="frame_length_ms",
            value=MATCH.frame_length_ms
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            parameter_name="hop_length_ms",
            value=MATCH.hop_length_ms
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            value=MATCH.window_function,
            status="resolved"
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_framing_windowing_shape(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        samples_per_channel,
        channels,
        batch_size,
        sample_rate,
        frame_length_ms,
        hop_length_ms,
        window_function
    ):
        samples_per_channel_int = int(samples_per_channel)
        channels_int = int(channels)
        frame_length_samples = (sample_rate * frame_length_ms) // 1000
        hop_length_samples = (sample_rate * hop_length_ms) // 1000
        num_frames = 1 + ((samples_per_channel_int - frame_length_samples) // hop_length_samples)

        input_shape = previous_output_shape
        output_shape = f"({batch_size}, {num_frames}, {frame_length_samples}, {channels_int})"
        parameter_shape = "None (framing and windowing has no learnable weights)"
        window_shape = f"({frame_length_samples},)"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="framing_windowing",
            weight_name="framing_windowing_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="input_audio_waveform",
            role="input",
            dims_count=3,
            shape=input_shape,
            data_type="float_audio_waveform_tensor",
            source=f"layer_{previous_layer_index}_audio_waveform"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="window_vector",
            role="intermediate",
            dims_count=1,
            shape=window_shape,
            data_type=f"float_{window_function}_window_vector",
            source="selected_window_function"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="framed_audio",
            role="output",
            dims_count=4,
            shape=output_shape,
            data_type="float_framed_audio_tensor",
            source="waveform_split_into_overlapping_windowed_frames"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="samples_per_channel",
            role="shape_metadata",
            dims_count=1,
            shape=str(samples_per_channel_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_samples_per_channel"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="output_num_frames",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_frames),
            data_type="integer_dimension_value",
            source="valid_framing_formula"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="output_frame_length",
            role="shape_metadata",
            dims_count=1,
            shape=str(frame_length_samples),
            data_type="integer_dimension_value",
            source="frame_length_ms_times_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="framing_windowing",
            tensor_name="output_channels",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_channel_count"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="framing_windowing",
            layer_kind="speech_preprocessing_layer",
            layer_role="audio_framing_and_windowing",
            repeat_policy="once_after_audio_input",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="framing_windowing",
            step_key=f"layer_{layer_index}_frame_length_samples",
            previous_step_key="none",
            step_order=1,
            title="Frame length in samples",
            formula="sample_rate × frame_length_ms / 1000",
            substitution=f"{sample_rate} × {frame_length_ms} / 1000",
            result=str(frame_length_samples)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="framing_windowing",
            step_key=f"layer_{layer_index}_hop_length_samples",
            previous_step_key=f"layer_{layer_index}_frame_length_samples",
            step_order=2,
            title="Hop length in samples",
            formula="sample_rate × hop_length_ms / 1000",
            substitution=f"{sample_rate} × {hop_length_ms} / 1000",
            result=str(hop_length_samples)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="framing_windowing",
            step_key=f"layer_{layer_index}_num_frames",
            previous_step_key=f"layer_{layer_index}_hop_length_samples",
            step_order=3,
            title="Number of frames",
            formula="1 + floor((samples_per_channel - frame_length_samples) / hop_length_samples)",
            substitution=f"1 + floor(({samples_per_channel_int} - {frame_length_samples}) / {hop_length_samples})",
            result=str(num_frames)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="framing_windowing",
            step_key=f"layer_{layer_index}_framing_output_shape",
            previous_step_key=f"layer_{layer_index}_num_frames",
            step_order=4,
            title="Framing + windowing output shape",
            formula="(batch_size, num_frames, frame_length_samples, channels)",
            substitution=f"({batch_size}, {num_frames}, {frame_length_samples}, {channels_int})",
            result=output_shape
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="framing_windowing",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
