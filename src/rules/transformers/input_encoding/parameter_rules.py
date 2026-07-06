from experta import Rule, MATCH, NOT

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    INPUT_ENCODING_SELECTED,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_VOCAB_SIZE,
    ASK_MODEL_DIM
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class InputEncodingParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="input_encoding"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="input_encoding",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_input_encoding_parameters(self, layer_index):
        print(INPUT_ENCODING_SELECTED)

        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        vocab_size = read_int_parameter(ASK_VOCAB_SIZE)
        model_dim = read_int_parameter(ASK_MODEL_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="input_encoding",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="input_encoding",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="input_encoding",
            parameter_name="vocab_size",
            value=vocab_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="input_encoding",
            parameter_name="model_dim",
            value=model_dim
        ))
