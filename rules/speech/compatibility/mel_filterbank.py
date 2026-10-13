from experta import Rule, MATCH, NOT, TEST, AS

from facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
    LayerParameter,
)
from prompts import ASK_MEL_MIN_FREQUENCY_HZ, ASK_MEL_MAX_FREQUENCY_HZ


from utils.console_control import read_int_parameter, read_text_parameter
class MelFilterbankCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="mel_filterbank"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank"
        )),
        salience=83
    )
    def rollback_mel_filterbank_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            from_shape="no previous STFT spectrum tensor",
            to_shape="(batch_size, num_frames, frequency_bins, channels)",
            method="Add Audio Input, Framing + Windowing, and STFT / Spectrum before Mel Filterbank."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="mel_filterbank",
            reason="mel_filterbank_needs_stft_spectrum",
            message=(
                "Mel Filterbank cannot be the first layer. "
                "It needs a magnitude or power spectrum from STFT / Spectrum with shape "
                "(batch_size, num_frames, frequency_bins, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mel_filterbank"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "magnitude_frequency_spectrum_tensor"),
        TEST(lambda data_type: data_type != "power_frequency_spectrum_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank"
        )),
        salience=82
    )
    def rollback_mel_filterbank_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, frequency_bins, channels)",
            method="Use STFT / Spectrum with magnitude or power output before Mel Filterbank."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mel_filterbank",
            reason="mel_filterbank_requires_real_spectrum_tensor",
            message=(
                "Mel Filterbank requires a magnitude or power frequency spectrum tensor. "
                "If STFT was created in complex mode, choose magnitude or power mode before adding Mel Filterbank."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mel_filterbank"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type == "magnitude_frequency_spectrum_tensor" or data_type == "power_frequency_spectrum_tensor"),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank"
        )),
        salience=82
    )
    def rollback_mel_filterbank_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, frequency_bins, channels)",
            method="Use STFT / Spectrum to produce a rank-4 spectrum tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="mel_filterbank",
            reason="mel_filterbank_requires_rank_4_spectrum",
            message=(
                "Mel Filterbank requires a rank-4 spectrum tensor with shape "
                "(batch_size, num_frames, frequency_bins, channels)."
            )
        ))

    @Rule(
        AS.max_frequency_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=MATCH.max_frequency_hz
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="mel_filterbank"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            layer_name="stft_spectrum",
            tensor_name="sample_rate",
            role="shape_metadata",
            shape=MATCH.sample_rate
        ),
        TEST(lambda max_frequency_hz, sample_rate: int(max_frequency_hz) > (int(sample_rate) // 2)),
        salience=67
    )
    def ask_max_frequency_again_if_above_nyquist(
        self,
        max_frequency_fact,
        layer_index,
        previous_layer_index,
        max_frequency_hz,
        sample_rate
    ):
        print()
        print(
            "Invalid maximum frequency. It must be less than or equal to the Nyquist frequency "
            f"({int(sample_rate) // 2} Hz)."
        )

        self.retract(max_frequency_fact)

        new_max_frequency_hz = read_int_parameter(ASK_MEL_MAX_FREQUENCY_HZ)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=new_max_frequency_hz
        ))

    @Rule(
        AS.max_frequency_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=MATCH.max_frequency_hz
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="mel_filterbank",
            parameter_name="min_frequency_hz",
            value=MATCH.min_frequency_hz
        ),
        TEST(lambda max_frequency_hz, min_frequency_hz: int(max_frequency_hz) <= int(min_frequency_hz)),
        salience=66
    )
    def ask_max_frequency_again_if_not_above_min(
        self,
        max_frequency_fact,
        layer_index,
        max_frequency_hz,
        min_frequency_hz
    ):
        print()
        print("Invalid Mel frequency range. Maximum frequency must be greater than minimum frequency.")

        self.retract(max_frequency_fact)

        new_max_frequency_hz = read_int_parameter(ASK_MEL_MAX_FREQUENCY_HZ)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="mel_filterbank",
            parameter_name="max_frequency_hz",
            value=new_max_frequency_hz
        ))
