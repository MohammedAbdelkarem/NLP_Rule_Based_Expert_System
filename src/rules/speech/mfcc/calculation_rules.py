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


class MFCCCalculationRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mfcc",
            layer_title=MATCH.layer_title
        ),
        Layer(
            layer_index=MATCH.previous_layer_index,
            output_shape=MATCH.previous_output_shape
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="log_mel_spectrogram",
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
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=MATCH.num_mfcc_coefficients
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_dct_type",
            value=MATCH.dct_type,
            status="resolved"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            question_key="mfcc_include_c0",
            value=MATCH.include_c0,
            status="resolved"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            parameter_name="lifter",
            value=MATCH.lifter
        ),
        NOT(Layer(layer_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_mfcc_shape(
        self,
        layer_index,
        previous_layer_index,
        layer_title,
        previous_output_shape,
        previous_data_type,
        sample_rate,
        batch_size,
        num_frames,
        num_mel_filters,
        channels,
        num_mfcc_coefficients,
        dct_type,
        include_c0,
        lifter
    ):
        batch_size_int = int(batch_size)
        num_frames_int = int(num_frames)
        num_mel_filters_int = int(num_mel_filters)
        channels_int = int(channels)
        sample_rate_int = int(sample_rate)
        num_mfcc_coefficients_int = int(num_mfcc_coefficients)
        lifter_int = int(lifter)

        input_shape = previous_output_shape
        output_shape = f"({batch_size_int}, {num_frames_int}, {num_mfcc_coefficients_int}, {channels_int})"
        parameter_shape = "None (MFCC has no learnable weights)"
        dct_matrix_shape = f"({num_mel_filters_int}, {num_mfcc_coefficients_int})"

        if include_c0 == "include_c0":
            coefficient_range = f"c0..c{num_mfcc_coefficients_int - 1}"
        else:
            coefficient_range = f"c1..c{num_mfcc_coefficients_int}"

        if lifter_int == 0:
            lifter_status = "disabled"
        else:
            lifter_status = f"enabled_lifter_{lifter_int}"

        self.declare(WeightShape(
            layer_index=layer_index,
            layer_name="mfcc",
            weight_name="mfcc_parameters",
            shape=parameter_shape
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="input_log_mel_spectrogram",
            role="input",
            dims_count=4,
            shape=input_shape,
            data_type=previous_data_type,
            source=f"layer_{previous_layer_index}_log_mel_spectrogram"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="fixed_dct_matrix",
            role="intermediate",
            dims_count=2,
            shape=dct_matrix_shape,
            data_type="fixed_float_dct_matrix_not_trainable",
            source="deterministic_discrete_cosine_transform_basis"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="mfcc_features",
            role="output",
            dims_count=4,
            shape=output_shape,
            data_type="mfcc_feature_tensor",
            source="fixed_dct_projection_over_log_mel_axis"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="sample_rate",
            role="shape_metadata",
            dims_count=1,
            shape=str(sample_rate_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_sample_rate"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=str(batch_size_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_batch_size"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="output_num_frames",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_frames_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_num_frames"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="input_mel_filters",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_mel_filters_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_mel_filters"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="output_mfcc_coefficients",
            role="shape_metadata",
            dims_count=1,
            shape=str(num_mfcc_coefficients_int),
            data_type="integer_dimension_value",
            source="user_selected_num_mfcc_coefficients"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="output_channels",
            role="shape_metadata",
            dims_count=1,
            shape=str(channels_int),
            data_type="integer_dimension_value",
            source=f"layer_{previous_layer_index}_channels"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="coefficient_range",
            role="shape_metadata",
            dims_count=1,
            shape=coefficient_range,
            data_type="mfcc_coefficient_selection_configuration",
            source="user_selected_include_c0_policy"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="dct_type",
            role="shape_metadata",
            dims_count=1,
            shape=dct_type,
            data_type="dct_type_configuration",
            source="user_selected_dct_type"
        ))

        self.declare(TensorInfo(
            layer_index=layer_index,
            layer_name="mfcc",
            tensor_name="lifter_status",
            role="shape_metadata",
            dims_count=1,
            shape=lifter_status,
            data_type="cepstral_lifter_configuration",
            source="user_selected_lifter"
        ))

        self.declare(ComponentGroup(
            group_index=layer_index,
            group_name="mfcc",
            group_title=layer_title,
            internal_layers=(
                "Log-Mel input validation + "
                "fixed DCT matrix construction + "
                "DCT projection over Mel axis + "
                "coefficient selection + "
                "optional cepstral liftering"
            ),
            combine_operation="fixed_discrete_cosine_transform_projection",
            input_shape=input_shape,
            output_shape=output_shape,
            status="computed"
        ))

        self.declare(LayerMetadata(
            layer_index=layer_index,
            layer_name="mfcc",
            layer_kind="speech_feature_extraction_component_group",
            layer_role="log_mel_spectrogram_to_mfcc_cepstral_projection",
            repeat_policy="once_after_log_mel_spectrogram",
            consecutive_repeat_policy="no_consecutive_repeat",
            terminal_status="non_terminal"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mfcc",
            step_key=f"layer_{layer_index}_mfcc_dct_matrix_shape",
            previous_step_key="none",
            step_order=1,
            title="Fixed DCT matrix shape",
            formula="(num_mel_filters, num_mfcc_coefficients)",
            substitution=f"({num_mel_filters_int}, {num_mfcc_coefficients_int})",
            result=dct_matrix_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mfcc",
            step_key=f"layer_{layer_index}_mfcc_output_shape",
            previous_step_key=f"layer_{layer_index}_mfcc_dct_matrix_shape",
            step_order=2,
            title="MFCC output shape",
            formula="(batch_size, num_frames, num_mfcc_coefficients, channels)",
            substitution=f"({batch_size_int}, {num_frames_int}, {num_mfcc_coefficients_int}, {channels_int})",
            result=output_shape
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mfcc",
            step_key=f"layer_{layer_index}_mfcc_coefficient_selection",
            previous_step_key=f"layer_{layer_index}_mfcc_output_shape",
            step_order=3,
            title="MFCC coefficient selection",
            formula="select first configured cepstral coefficients after DCT",
            substitution=f"include_c0={include_c0}, coefficient_range={coefficient_range}",
            result=f"output_feature_dimension={num_mfcc_coefficients_int}"
        ))

        self.declare(CalculationStep(
            layer_index=layer_index,
            layer_name="mfcc",
            step_key=f"layer_{layer_index}_mfcc_parameter_status",
            previous_step_key=f"layer_{layer_index}_mfcc_coefficient_selection",
            step_order=4,
            title="Trainable parameter status",
            formula="MFCC uses a fixed DCT matrix, not learned weights",
            substitution=f"dct_type={dct_type}, dct_matrix_shape={dct_matrix_shape}, lifter={lifter_status}",
            result="trainable=False, has_weights=False, has_matrix=True"
        ))

        self.declare(Layer(
            layer_index=layer_index,
            layer_name="mfcc",
            layer_title=layer_title,
            input_shape=input_shape,
            output_shape=output_shape
        ))
