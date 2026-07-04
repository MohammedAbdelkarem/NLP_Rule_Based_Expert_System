from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    WeightShape,
    Layer,
    ComponentGroup,
    LayerMetadata,
    CalculationStep,
)


class STFTSpectrumCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="stft_spectrum",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_num_frames",
            role="shape_metadata",
            shape=MATCH.num_frames
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_frame_length",
            role="shape_metadata",
            shape=MATCH.frame_length_samples
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_channels",
            role="shape_metadata",
            shape=MATCH.channels
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="sample_rate",
            role="shape_metadata",
            shape=MATCH.sample_rate
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            parameter_name="n_fft",
            value=MATCH.n_fft
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            question_key="stft_output_mode",
            value=MATCH.output_mode,
            status="resolved"
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_stft_spectrum_shape(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        batch_size,
        num_frames,
        frame_length_samples,
        channels,
        sample_rate,
        n_fft,
        output_mode
    ):
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        frame_length_samples_int = int(frame_length_samples)
        channels_int = int(channels)
        n_fft_int = int(n_fft)
        frequency_bins = (n_fft_int // 2) + 1

        input_shape = previous_output_shape
        output_shape = f"({batch_size_int}, {num_frames_int}, {frequency_bins}, {channels_int})"
        parameter_shape = "None (STFT / Spectrum extraction has no learnable weights)"
        stft_kernel_shape = f"fixed Fourier basis ({frequency_bins}, {n_fft_int})"
        output_data_type = f"{output_mode}_frequency_spectrum_tensor"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            weight_name="stft_spectrum_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="input_framed_audio",
            role="input",
            dims_count=4,
            shape=input_shape,
            data_type="float_framed_audio_tensor",
            source=f"layer_{previous_layer_index}_framed_audio"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="fixed_fourier_basis",
            role="intermediate",
            dims_count=2,
            shape=stft_kernel_shape,
            data_type="fixed_complex_fourier_basis_not_trainable",
            source="n_fft_configuration"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="frequency_spectrum",
            role="output",
            dims_count=4,
            shape=output_shape,
            data_type=output_data_type,
            source="short_time_fourier_transform_over_each_frame"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="output_num_frames",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_frames_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_num_frames"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="output_frequency_bins",
            role="shape_metadata",
            dims_count=1,
            shape=str(frequency_bins),
            data_type="integer_dimension_value",
            source="n_fft_divided_by_two_plus_one"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            tensor_name="output_channels",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_channels"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="stft_spectrum",
            group_title=layer_title,
            internal_layers=(
                "Frame length check / zero padding + "
                "fixed FFT projection + "
                "positive frequency bin selection + "
                "magnitude/power/complex output conversion"
            ),
            combine_operation="fixed_short_time_fourier_transform",
            input_shape=input_shape,
            output_shape=output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            layer_kind="speech_feature_extraction_component_group",
            layer_role="short_time_frequency_spectrum_extraction",
            repeat_policy="once_after_framing_windowing",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            step_key=f"layer_{layer_index}_frequency_bins",
            previous_step_key="none",
            step_order=1,
            title="Frequency bins",
            formula="floor(n_fft / 2) + 1",
            substitution=f"floor({n_fft_int} / 2) + 1",
            result=str(frequency_bins)
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            step_key=f"layer_{layer_index}_stft_output_shape",
            previous_step_key=f"layer_{layer_index}_frequency_bins",
            step_order=2,
            title="STFT / Spectrum output shape",
            formula="(batch_size, num_frames, frequency_bins, channels)",
            substitution=f"({batch_size_int}, {num_frames_int}, {frequency_bins}, {channels_int})",
            result=output_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            step_key=f"layer_{layer_index}_stft_parameter_status",
            previous_step_key=f"layer_{layer_index}_stft_output_shape",
            step_order=3,
            title="Trainable parameter status",
            formula="STFT uses a fixed Fourier transform, not learned weights",
            substitution=f"n_fft={n_fft_int}, output_mode={output_mode}",
            result="trainable=False, has_weights=False"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
