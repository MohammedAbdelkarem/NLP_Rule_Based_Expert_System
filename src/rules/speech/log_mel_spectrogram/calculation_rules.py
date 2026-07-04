from experta import Rule, MATCH, NOT

from src.facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    WeightShape,
    Layer,
    LayerMetadata,
    CalculationStep,
)


class LogMelSpectrogramCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="log_mel_spectrogram",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="mel_spectrum",
            role="output",
            dims_count=4,
            shape=MATCH.previous_output_shape,
            data_type="mel_frequency_spectrum_tensor"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="sample_rate",
            role="shape_metadata",
            shape=MATCH.sample_rate
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
            tensor_name="output_mel_filters",
            role="shape_metadata",
            shape=MATCH.num_mel_filters
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_channels",
            role="shape_metadata",
            shape=MATCH.channels
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            parameter_name="epsilon_power",
            value=MATCH.epsilon_power
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram",
            question_key="log_mel_compression_mode",
            value=MATCH.compression_mode,
            status="resolved"
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_log_mel_spectrogram_shape(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        sample_rate,
        batch_size,
        num_frames,
        num_mel_filters,
        channels,
        epsilon_power,
        compression_mode
    ):
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        num_mel_filters_int = int(num_mel_filters)
        channels_int = int(channels)
        sample_rate_int = int(sample_rate)
        epsilon_power_int = int(epsilon_power)
        epsilon_text = f"1e-{epsilon_power_int}"

        input_shape = previous_output_shape
        output_shape = f"({batch_size_int}, {num_frames_int}, {num_mel_filters_int}, {channels_int})"
        parameter_shape = "None (Log-Mel Spectrogram has no learnable weights)"

        if compression_mode == "decibel":
            formula = "10 * log10(mel_energy + epsilon)"
            output_data_type = "log_mel_decibel_spectrogram_tensor"
        elif compression_mode == "log10":
            formula = "log10(mel_energy + epsilon)"
            output_data_type = "log10_mel_spectrogram_tensor"
        else:
            formula = "log(mel_energy + epsilon)"
            output_data_type = "log_mel_spectrogram_tensor"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            weight_name="log_mel_spectrogram_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="input_mel_spectrum",
            role="input",
            dims_count=4,
            shape=input_shape,
            data_type="mel_frequency_spectrum_tensor",
            source=f"layer_{previous_layer_index}_mel_spectrum"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="epsilon",
            role="shape_metadata",
            dims_count=1,
            shape=epsilon_text,
            data_type="positive_float_scalar_configuration",
            source="user_selected_epsilon"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="log_mel_spectrogram",
            role="output",
            dims_count=4,
            shape=output_shape,
            data_type=output_data_type,
            source="elementwise_log_compression_over_mel_energy"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="output_num_frames",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_frames_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_num_frames"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="output_mel_filters",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_mel_filters_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_mel_filters"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="output_channels",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_channels"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            tensor_name="compression_mode",
            role="shape_metadata",
            dims_count=1,
            shape=compression_mode,
            data_type="log_compression_mode_configuration",
            source="user_selected_log_compression_mode"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            layer_kind="speech_feature_extraction_layer",
            layer_role="elementwise_log_compression_for_mel_features",
            repeat_policy="once_after_mel_filterbank",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            step_key=f"layer_{layer_index}_log_mel_shape_preservation",
            previous_step_key="none",
            step_order=1,
            title="Log-Mel shape preservation",
            formula="output_shape = input_shape",
            substitution=f"output_shape = {input_shape}",
            result=output_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            step_key=f"layer_{layer_index}_log_mel_compression",
            previous_step_key=f"layer_{layer_index}_log_mel_shape_preservation",
            step_order=2,
            title="Element-wise log compression",
            formula=formula,
            substitution=f"epsilon={epsilon_text}, compression_mode={compression_mode}",
            result="same_shape_log_compressed_mel_features"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            step_key=f"layer_{layer_index}_log_mel_parameter_status",
            previous_step_key=f"layer_{layer_index}_log_mel_compression",
            step_order=3,
            title="Trainable parameter status",
            formula="Log-Mel Spectrogram applies an element-wise fixed mathematical function",
            substitution=f"epsilon={epsilon_text}, compression_mode={compression_mode}",
            result="trainable=False, has_weights=False, has_matrix=False"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
