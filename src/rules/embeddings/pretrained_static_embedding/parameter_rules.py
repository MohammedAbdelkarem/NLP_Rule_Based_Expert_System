from experta import MATCH, NOT, Rule

from src.facts import LayerParameter, OptionSelection, SelectedLayer
from src.prompts import (
    ASK_BATCH_SIZE,
    ASK_EMBEDDING_DIM,
    ASK_PRETRAINED_EMBEDDING_SOURCE,
    ASK_PRETRAINED_EMBEDDING_TRAINING_MODE,
    ASK_SEQUENCE_LENGTH,
    ASK_VOCAB_SIZE,
    PRETRAINED_STATIC_EMBEDDING_SELECTED,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class PretrainedStaticEmbeddingParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="pretrained_static_embedding",
                parameter_name="batch_size",
            ),
        ),
        salience=76,
    )
    def ask_pretrained_static_embedding_shape_parameters(self, layer_index):
        print(PRETRAINED_STATIC_EMBEDDING_SELECTED)

        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        vocab_size = read_int_parameter(ASK_VOCAB_SIZE)
        embedding_dim = read_int_parameter(ASK_EMBEDDING_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="batch_size",
            value=batch_size,
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="sequence_length",
            value=sequence_length,
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="vocab_size",
            value=vocab_size,
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="embedding_dim",
            value=embedding_dim,
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            parameter_name="embedding_dim",
            value=MATCH.embedding_dim,
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="pretrained_static_embedding",
                question_key="pretrained_embedding_source",
            ),
        ),
        salience=75,
    )
    def ask_pretrained_embedding_source(self, layer_index, embedding_dim):
        choice = read_text_parameter(ASK_PRETRAINED_EMBEDDING_SOURCE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="pretrained_static_embedding",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            status="resolved",
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="pretrained_static_embedding",
                question_key="pretrained_embedding_training_mode",
            ),
        ),
        salience=72,
    )
    def ask_pretrained_training_mode(self, layer_index):
        choice = read_text_parameter(ASK_PRETRAINED_EMBEDDING_TRAINING_MODE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice=choice,
            value="unknown",
            status="raw",
        ))
