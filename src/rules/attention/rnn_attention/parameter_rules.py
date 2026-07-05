from experta import MATCH, NOT, Rule, TEST, AS

from src.facts import LayerParameter, OptionSelection, SelectedLayer
from src.prompts import (
    ASK_ATTENTION_SCORING_TYPE,
    ASK_BATCH_SIZE,
    ASK_DECODER_HIDDEN_SIZE,
    ASK_ENCODER_HIDDEN_SIZE,
    ASK_SOURCE_SEQUENCE_LENGTH,
    ASK_TARGET_SEQUENCE_LENGTH,
    RNN_ATTENTION_SELECTED,
)


class RNNAttentionParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, previous_layer_index=0, layer_name="rnn_attention"),
        NOT(LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size")),
        salience=72
    )
    def ask_rnn_attention_parameters(self, layer_index):
        print(RNN_ATTENTION_SELECTED)
        scoring_choice = input(ASK_ATTENTION_SCORING_TYPE)
        batch_size = int(input(ASK_BATCH_SIZE))
        source_sequence_length = int(input(ASK_SOURCE_SEQUENCE_LENGTH))
        target_sequence_length = int(input(ASK_TARGET_SEQUENCE_LENGTH))
        encoder_hidden_size = int(input(ASK_ENCODER_HIDDEN_SIZE))
        decoder_hidden_size = int(input(ASK_DECODER_HIDDEN_SIZE))
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice=scoring_choice, value="unknown", status="raw"))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_attention", parameter_name="batch_size", value=batch_size))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_attention", parameter_name="source_sequence_length", value=source_sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_attention", parameter_name="target_sequence_length", value=target_sequence_length))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size", value=encoder_hidden_size))
        self.declare(LayerParameter(layer_index=layer_index, layer_name="rnn_attention", parameter_name="decoder_hidden_size", value=decoder_hidden_size))

    @Rule(
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size", value=MATCH.hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="decoder_hidden_size", value=MATCH.hidden_size),
        NOT(OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_hidden_match")),
        salience=71
    )
    def declare_attention_hidden_match(self, layer_index):
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_hidden_match", choice="same", value="match", status="resolved"))

    @Rule(
        AS.encoder_fact << LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="encoder_hidden_size", value=MATCH.encoder_hidden_size),
        LayerParameter(layer_index=MATCH.layer_index, layer_name="rnn_attention", parameter_name="decoder_hidden_size", value=MATCH.decoder_hidden_size),
        TEST(lambda encoder_hidden_size, decoder_hidden_size: int(encoder_hidden_size) != int(decoder_hidden_size)),
        NOT(OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_hidden_match")),
        salience=71
    )
    def declare_attention_hidden_mismatch(self, layer_index):
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_hidden_match", choice="different", value="mismatch", status="resolved"))
