from experta import MATCH, NOT, Rule

from facts import ComponentGroup, LayerParameter, SelectedLayer
from rules.sequence_to_sequence.rnn_encoder_decoder.common import (
    declare_encoder_decoder_group,
    encoder_decoder_weight_summary,
    shape_3d,
)


class RNNEncoderDecoderFirstCalculationRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_encoder_decoder", layer_title=MATCH.layer_title),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="batch_size", value=MATCH.batch_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="source_sequence_length", value=MATCH.source_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="input_dim", value=MATCH.input_dim),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="target_sequence_length", value=MATCH.target_sequence_length),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="hidden_size", value=MATCH.hidden_size),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        salience=60
    )
    def calculate_first_rnn_encoder_decoder(self, layer_index, layer_title, batch_size, source_sequence_length, input_dim, target_sequence_length, hidden_size):
        input_shape = shape_3d(batch_size, source_sequence_length, input_dim)
        output_shape = shape_3d(batch_size, target_sequence_length, hidden_size)
        parameter_shape = encoder_decoder_weight_summary(input_dim, hidden_size)
        declare_encoder_decoder_group(self, layer_index, layer_title, input_shape, output_shape, batch_size, source_sequence_length, target_sequence_length, input_dim, hidden_size, parameter_shape)
