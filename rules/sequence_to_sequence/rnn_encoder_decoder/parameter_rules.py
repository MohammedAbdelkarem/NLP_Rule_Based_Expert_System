from experta import MATCH, NOT, Rule

from facts import LayerParameter, SelectedLayer
from prompts import (
    ASK_BATCH_SIZE,
    ASK_HIDDEN_SIZE,
    ASK_INPUT_DIM,
    ASK_SOURCE_SEQUENCE_LENGTH,
    ASK_TARGET_SEQUENCE_LENGTH,
    RNN_ENCODER_DECODER_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class RNNEncoderDecoderParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_encoder_decoder"),
        NOT(LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="hidden_size")),
        salience=72
    )
    def ask_first_rnn_encoder_decoder_parameters(self, layer_index):
        print(RNN_ENCODER_DECODER_SELECTED)
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        source_sequence_length = read_int_parameter(ASK_SOURCE_SEQUENCE_LENGTH)
        input_dim = read_int_parameter(ASK_INPUT_DIM)
        target_sequence_length = read_int_parameter(ASK_TARGET_SEQUENCE_LENGTH)
        hidden_size = read_int_parameter(ASK_HIDDEN_SIZE)
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="batch_size", value=batch_size))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="source_sequence_length", value=source_sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="input_dim", value=input_dim))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="target_sequence_length", value=target_sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="hidden_size", value=hidden_size))

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=MATCH.previous_layer_index, layer_name="rnn_encoder_decoder"),
        NOT(LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_encoder_decoder", parameter_name="hidden_size")),
        NOT(SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_encoder_decoder")),
        salience=72
    )
    def ask_next_rnn_encoder_decoder_parameters(self, layer_index):
        print(RNN_ENCODER_DECODER_SELECTED)
        target_sequence_length = read_int_parameter(ASK_TARGET_SEQUENCE_LENGTH)
        hidden_size = read_int_parameter(ASK_HIDDEN_SIZE)
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="target_sequence_length", value=target_sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_encoder_decoder", parameter_name="hidden_size", value=hidden_size))
