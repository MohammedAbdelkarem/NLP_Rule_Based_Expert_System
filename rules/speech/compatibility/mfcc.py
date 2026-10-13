from experta import Rule, MATCH, NOT, TEST, AS

from facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
    LayerParameter,
)
from prompts import ASK_NUM_MFCC_COEFFICIENTS


from utils.console_control import read_int_parameter, read_text_parameter
class MFCCCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="mfcc"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mfcc"
        )),
        salience=83
    )
    def rollback_mfcc_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mfcc",
            from_shape="no previous Log-Mel spectrogram tensor",
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Add Audio Input, Framing + Windowing, STFT / Spectrum, Mel Filterbank, and Log-Mel Spectrogram before MFCC."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="mfcc",
            reason="mfcc_needs_log_mel_spectrogram_output",
            message=(
                "MFCC cannot be the first layer. "
                "It needs Log-Mel Spectrogram output with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mfcc"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "log_mel_spectrogram_tensor"),
        TEST(lambda data_type: data_type != "log10_mel_spectrogram_tensor"),
        TEST(lambda data_type: data_type != "log_mel_decibel_spectrogram_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mfcc"
        )),
        salience=82
    )
    def rollback_mfcc_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mfcc",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Log-Mel Spectrogram output before MFCC."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mfcc",
            reason="mfcc_requires_log_mel_spectrogram_tensor",
            message=(
                "MFCC requires a Log-Mel spectrogram tensor. "
                "Add Log-Mel Spectrogram before this component group."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mfcc"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type == "log_mel_spectrogram_tensor" or data_type == "log10_mel_spectrogram_tensor" or data_type == "log_mel_decibel_spectrogram_tensor"),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mfcc"
        )),
        salience=82
    )
    def rollback_mfcc_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mfcc",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Log-Mel Spectrogram to produce a rank-4 feature tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mfcc",
            reason="mfcc_requires_rank_4_log_mel_spectrogram",
            message=(
                "MFCC requires a rank-4 Log-Mel spectrogram tensor with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))

    @Rule(
        AS.num_mfcc_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=MATCH.num_mfcc_coefficients
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mfcc"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_mel_filters",
            role="shape_metadata",
            shape=MATCH.num_mel_filters
        ),
        TEST(lambda num_mfcc_coefficients, num_mel_filters: int(num_mfcc_coefficients) > int(num_mel_filters)),
        salience=67
    )
    def ask_num_mfcc_again_if_above_mel_filters(
        self,
        num_mfcc_fact,
        layer_index,
        previous_layer_index,
        num_mfcc_coefficients,
        num_mel_filters
    ):
        print()
        print(
            "Invalid number of MFCC coefficients. It must be less than or equal to "
            f"the number of Mel filters ({num_mel_filters})."
        )

        self.retract(num_mfcc_fact)

        new_num_mfcc_coefficients = read_int_parameter(ASK_NUM_MFCC_COEFFICIENTS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mfcc",
            parameter_name="num_mfcc_coefficients",
            value=new_num_mfcc_coefficients
        ))
