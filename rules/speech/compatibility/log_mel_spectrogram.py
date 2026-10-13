from experta import Rule, MATCH, NOT, TEST

from facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
)


class LogMelSpectrogramCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="log_mel_spectrogram"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram"
        )),
        salience=83
    )
    def rollback_log_mel_spectrogram_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            from_shape="no previous Mel spectrum tensor",
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Add Audio Input, Framing + Windowing, STFT / Spectrum, and Mel Filterbank before Log-Mel Spectrogram."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="log_mel_spectrogram",
            reason="log_mel_spectrogram_needs_mel_filterbank_output",
            message=(
                "Log-Mel Spectrogram cannot be the first layer. "
                "It needs Mel Filterbank output with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="log_mel_spectrogram"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "mel_frequency_spectrum_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram"
        )),
        salience=82
    )
    def rollback_log_mel_spectrogram_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Mel Filterbank output before Log-Mel Spectrogram."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrogram",
            reason="log_mel_spectrogram_requires_mel_spectrum_tensor",
            message=(
                "Log-Mel Spectrogram requires a Mel frequency spectrum tensor. "
                "Add Mel Filterbank before this layer."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="log_mel_spectrogram"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="mel_frequency_spectrum_tensor"
        ),
        TEST(lambda dims_count: dims_count != 4),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrogram"
        )),
        salience=82
    )
    def rollback_log_mel_spectrogram_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrogram",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Mel Filterbank to produce a rank-4 Mel spectrum tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrogram",
            reason="log_mel_spectrogram_requires_rank_4_mel_spectrum",
            message=(
                "Log-Mel Spectrogram requires a rank-4 Mel spectrum tensor with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))
