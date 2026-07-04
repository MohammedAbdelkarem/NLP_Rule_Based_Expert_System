from experta import Rule, MATCH, NOT, TEST

from src.facts import (
    SelectedLayer,
    RollbackRequest,
    TensorInfo,
    TransformationSuggestion,
)


class LogMelSpectrumCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="log_mel_spectrum"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="log_mel_spectrum"
        )),
        salience=83
    )
    def rollback_log_mel_spectrum_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrum",
            from_shape="no previous Mel spectrum tensor",
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Add Audio Input, Framing + Windowing, STFT / Spectrum, and Mel Filterbank before Log-Mel Spectrum."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="log_mel_spectrum",
            reason="log_mel_spectrum_needs_mel_filterbank_output",
            message=(
                "Log-Mel Spectrum cannot be the first layer. "
                "It needs Mel Filterbank output with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="log_mel_spectrum"
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
            layer_name="log_mel_spectrum"
        )),
        salience=82
    )
    def rollback_log_mel_spectrum_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrum",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Mel Filterbank output before Log-Mel Spectrum."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrum",
            reason="log_mel_spectrum_requires_mel_spectrum_tensor",
            message=(
                "Log-Mel Spectrum requires a Mel frequency spectrum tensor. "
                "Add Mel Filterbank before this layer."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="log_mel_spectrum"
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
            layer_name="log_mel_spectrum"
        )),
        salience=82
    )
    def rollback_log_mel_spectrum_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="log_mel_spectrum",
            from_shape=previous_shape,
            to_shape="(batch_size, num_frames, num_mel_filters, channels)",
            method="Use Mel Filterbank to produce a rank-4 Mel spectrum tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="log_mel_spectrum",
            reason="log_mel_spectrum_requires_rank_4_mel_spectrum",
            message=(
                "Log-Mel Spectrum requires a rank-4 Mel spectrum tensor with shape "
                "(batch_size, num_frames, num_mel_filters, channels)."
            )
        ))
