from experta import Rule, MATCH, NOT, TEST

from facts import SelectedLayer, RollbackRequest, TensorInfo, TransformationSuggestion, LayerParameter


class FramingWindowingCompatibilityRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="framing_windowing"
        ),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=83
    )
    def rollback_framing_windowing_as_first_layer(self, layer_index):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="framing_windowing",
            from_shape="no previous waveform tensor",
            to_shape="(batch_size, samples_per_channel, channels)",
            method="Add Audio Input Layer before Framing + Windowing."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=0,
            layer_name="framing_windowing",
            reason="framing_windowing_needs_audio_waveform",
            message=(
                "Framing + Windowing cannot be the first layer. "
                "It needs an audio waveform tensor from Audio Input with shape "
                "(batch_size, samples_per_channel, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda data_type: data_type != "float_audio_waveform_tensor"),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=82
    )
    def rollback_framing_windowing_for_invalid_previous_type(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape,
        data_type
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="framing_windowing",
            from_shape=previous_shape,
            to_shape="(batch_size, samples_per_channel, channels)",
            method="Use Audio Input before Framing + Windowing or choose a layer compatible with the previous tensor."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            reason="framing_windowing_requires_waveform_tensor",
            message=(
                "Framing + Windowing requires a float audio waveform tensor. "
                "The previous layer output is not raw waveform audio."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.previous_shape,
            data_type="float_audio_waveform_tensor"
        ),
        TEST(lambda dims_count: dims_count != 3),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=82
    )
    def rollback_framing_windowing_for_invalid_previous_rank(
        self,
        layer_index,
        previous_layer_index,
        dims_count,
        previous_shape
    ):
        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="framing_windowing",
            from_shape=previous_shape,
            to_shape="(batch_size, samples_per_channel, channels)",
            method="Use Audio Input to produce a rank-3 waveform tensor before Framing + Windowing."
        ))

        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            reason="framing_windowing_requires_rank_3_waveform",
            message=(
                "Framing + Windowing requires rank-3 waveform input with shape "
                "(batch_size, samples_per_channel, channels)."
            )
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing"
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
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="samples_per_channel",
            role="shape_metadata",
            shape=MATCH.samples_per_channel
        ),
        LayerParameter(
            layer_index=MATCH.previous_layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=MATCH.sample_rate
        ),
        TEST(lambda frame_length_ms: frame_length_ms <= 0),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=81
    )
    def rollback_invalid_frame_length_zero(
        self,
        layer_index,
        previous_layer_index,
        frame_length_ms,
        hop_length_ms,
        samples_per_channel,
        sample_rate
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            reason="invalid_frame_length",
            message="Frame length must be greater than 0 ms. Choose the layer again with a valid frame length."
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing"
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
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="samples_per_channel",
            role="shape_metadata",
            shape=MATCH.samples_per_channel
        ),
        LayerParameter(
            layer_index=MATCH.previous_layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=MATCH.sample_rate
        ),
        TEST(lambda hop_length_ms: hop_length_ms <= 0),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=81
    )
    def rollback_invalid_hop_length_zero(
        self,
        layer_index,
        previous_layer_index,
        frame_length_ms,
        hop_length_ms,
        samples_per_channel,
        sample_rate
    ):
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            reason="invalid_hop_length",
            message="Hop length must be greater than 0 ms. Choose the layer again with a valid hop length."
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="framing_windowing"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            parameter_name="frame_length_ms",
            value=MATCH.frame_length_ms
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="samples_per_channel",
            role="shape_metadata",
            shape=MATCH.samples_per_channel
        ),
        LayerParameter(
            layer_index=MATCH.previous_layer_index,
            layer_name="audio_input",
            parameter_name="sample_rate",
            value=MATCH.sample_rate
        ),
        TEST(lambda frame_length_ms, sample_rate, samples_per_channel: ((sample_rate * frame_length_ms) // 1000) > int(samples_per_channel)),
        NOT(RollbackRequest(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        )),
        salience=81
    )
    def rollback_frame_longer_than_audio(
        self,
        layer_index,
        previous_layer_index,
        frame_length_ms,
        samples_per_channel,
        sample_rate
    ):
        frame_length_samples = (sample_rate * frame_length_ms) // 1000
        self.declare(RollbackRequest(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="framing_windowing",
            reason="frame_longer_than_audio",
            message=(
                "Frame length is longer than the available audio samples. "
                f"Frame length samples = {frame_length_samples}, audio samples per channel = {samples_per_channel}."
            )
        ))
