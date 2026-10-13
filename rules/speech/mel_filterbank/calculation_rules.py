from experta import Rule, MATCH, NOT

from facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    WeightShape,
    Layer,
    ComponentGroup,
    LayerMetadata,
    CalculationStep,
)


class MelFilterbankCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mel_filterbank",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="frequency_spectrum",
            role="output",
            dims_count=4,
            shape=MATCH.previous_output_shape,
            data_type=MATCH.previous_data_type
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
            tensor_name="output_frequency_bins",
            role="shape_metadata",
            shape=MATCH.frequency_bins
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_channels",
            role="shape_metadata",
            shape=MATCH.channels
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="num_mel_filters",
            value=MATCH.num_mel_filters
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="min_frequency_hz",
            value=MATCH.min_frequency_hz
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=MATCH.max_frequency_hz
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_mel_filterbank_shape(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        previous_data_type,
        sample_rate,
        batch_size,
        num_frames,
        frequency_bins,
        channels,
        num_mel_filters,
        min_frequency_hz,
        max_frequency_hz
    ):
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        frequency_bins_int = int(frequency_bins)
        channels_int = int(channels)
        sample_rate_int = int(sample_rate)
        num_mel_filters_int = int(num_mel_filters)
        min_frequency_hz_int = int(min_frequency_hz)
        max_frequency_hz_int = int(max_frequency_hz)
        nyquist_frequency_hz = sample_rate_int // 2

        input_shape = previous_output_shape
        output_shape = f"({batch_size_int}, {num_frames_int}, {num_mel_filters_int}, {channels_int})"
        parameter_shape = "None (Mel filterbank has no learnable weights)"
        matrix_shape = f"({frequency_bins_int}, {num_mel_filters_int})"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            weight_name="mel_filterbank_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="input_frequency_spectrum",
            role="input",
            dims_count=4,
            shape=input_shape,
            data_type=previous_data_type,
            source=f"layer_{previous_layer_index}_frequency_spectrum"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="fixed_mel_filterbank_matrix",
            role="intermediate",
            dims_count=2,
            shape=matrix_shape,
            data_type="fixed_float_mel_filterbank_matrix_not_trainable",
            source="triangular_mel_filters_from_frequency_range"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="mel_spectrum",
            role="output",
            dims_count=4,
            shape=output_shape,
            data_type="mel_frequency_spectrum_tensor",
            source="fixed_matrix_projection_from_linear_frequency_to_mel_bands"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="nyquist_frequency_hz",
            role="shape_metadata",
            dims_count=1,
            shape=str(nyquist_frequency_hz),
            data_type="integer_dimension_value",
            source="sample_rate_divided_by_two"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="output_num_frames",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_frames_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_num_frames"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="input_frequency_bins",
            role="shape_metadata",
            dims_count=1,
            shape=str(frequency_bins_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_frequency_bins"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="output_mel_filters",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_mel_filters_int),
            data_type="integer_dimension_value",
            source="user_selected_num_mel_filters"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="output_channels",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_channels"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            tensor_name="mel_frequency_range_hz",
            role="shape_metadata",
            dims_count=1,
            shape=f"{min_frequency_hz_int}..{max_frequency_hz_int}",
            data_type="frequency_range_hz",
            source="user_selected_mel_frequency_range"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="mel_filterbank",
            group_title=layer_title,
            internal_layers=(
                "frequency range validation + "
                "Hz-to-Mel conversion + "
                "Mel filter boundary calculation + "
                "triangular filterbank matrix construction + "
                "fixed spectrum-to-Mel matrix projection"
            ),
            combine_operation="fixed_mel_filterbank_projection",
            input_shape=input_shape,
            output_shape=output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            layer_kind="speech_feature_extraction_component_group",
            layer_role="linear_frequency_to_mel_frequency_projection",
            repeat_policy="once_after_stft_spectrum",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            step_key=f"layer_{layer_index}_mel_matrix_shape",
            previous_step_key="none",
            step_order=1,
            title="Mel filterbank matrix shape",
            formula="(frequency_bins, num_mel_filters)",
            substitution=f"({frequency_bins_int}, {num_mel_filters_int})",
            result=matrix_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            step_key=f"layer_{layer_index}_mel_output_shape",
            previous_step_key=f"layer_{layer_index}_mel_matrix_shape",
            step_order=2,
            title="Mel filterbank output shape",
            formula="(batch_size, num_frames, num_mel_filters, channels)",
            substitution=f"({batch_size_int}, {num_frames_int}, {num_mel_filters_int}, {channels_int})",
            result=output_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            step_key=f"layer_{layer_index}_mel_parameter_status",
            previous_step_key=f"layer_{layer_index}_mel_output_shape",
            step_order=3,
            title="Trainable parameter status",
            formula="Mel filterbank uses a fixed triangular filter matrix, not learned weights",
            substitution=(
                f"frequency_range={min_frequency_hz_int}..{max_frequency_hz_int} Hz, "
                f"matrix_shape={matrix_shape}"
            ),
            result="trainable=False, has_weights=False, has_matrix=True"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
