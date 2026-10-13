from experta import Rule, MATCH, NOT, TEST, AS

from facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
    LayerParameter,
)
from prompts import ASK_STFT_N_FFT


from utils.console_control import read_int_parameter, read_text_parameter
class STFTSpectrumCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="stft_spectrum"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum"
        )),
        salience=83
    )
    def rollback_stft_spectrum_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            from_shape="no previous framed audio tensor",
            to_shape="(batch_size, num_frames, frame_length_samples, channels)",
            method="Add Audio Input, then Framing + Windowing, before STFT / Spectrum."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="stft_spectrum",
            reason="stft_spectrum_needs_framed_audio",
            message=(
                "STFT / Spectrum cannot be the first layer. "
                "It needs framed audio from Framing + Windowing with shape "
                "(batch_size, num_frames, frame_length_samples, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="stft_spectrum"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "float_framed_audio_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum"
        )),
        salience=82
    )
    def rollback_stft_spectrum_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, frame_length_samples, channels)",
            method="Use Framing + Windowing before STFT / Spectrum."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="stft_spectrum",
            reason="stft_spectrum_requires_framed_audio_tensor",
            message=(
                "STFT / Spectrum requires a framed audio tensor. "
                "The previous layer output is not framed audio."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="stft_spectrum"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="float_framed_audio_tensor"
        ),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum"
        )),
        salience=82
    )
    def rollback_stft_spectrum_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, frame_length_samples, channels)",
            method="Use Framing + Windowing to produce a rank-4 framed audio tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="stft_spectrum",
            reason="stft_spectrum_requires_rank_4_framed_audio",
            message=(
                "STFT / Spectrum requires a rank-4 framed audio tensor with shape "
                "(batch_size, num_frames, frame_length_samples, channels)."
            )
        ))

    @Rule(
        AS.n_fft_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="stft_spectrum",
            parameter_name="n_fft",
            value=MATCH.n_fft
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="stft_spectrum"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing",
            tensor_name="output_frame_length",
            role="shape_metadata",
            shape=MATCH.frame_length_samples
        ),
        TEST(lambda n_fft, frame_length_samples: int(n_fft) < int(frame_length_samples)),
        salience=67
    )
    def ask_n_fft_again_if_smaller_than_frame_length(
        self,
        n_fft_fact,
        layer_index,
        previous_layer_index,
        n_fft,
        frame_length_samples
    ):
        print()
        print(
            "Invalid n_fft. It must be greater than or equal to the frame length "
            f"({frame_length_samples} samples)."
        )

        self.retract(n_fft_fact)

        new_n_fft = read_int_parameter(ASK_STFT_N_FFT)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="stft_spectrum",
            parameter_name="n_fft",
            value=new_n_fft
        ))
