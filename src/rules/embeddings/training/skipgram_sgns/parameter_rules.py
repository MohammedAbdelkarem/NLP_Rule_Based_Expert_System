from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    SGNS_SELECTED,
    ASK_TARGET_VOCAB_SIZE,
    ASK_CONTEXT_VOCAB_SIZE,
    ASK_EMBEDDING_DIM,
    ASK_CONTEXT_WINDOW_SIZE,
    ASK_NEGATIVE_SAMPLE_COUNT,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class SkipGramSGNSParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name="target_vocab_size"
        )),
        salience=70
    )
    def ask_skipgram_sgns_parameters(self, layer_index):
        print(SGNS_SELECTED)

        target_vocab_size = read_int_parameter(ASK_TARGET_VOCAB_SIZE)
        context_vocab_size = read_int_parameter(ASK_CONTEXT_VOCAB_SIZE)
        embedding_dim = read_int_parameter(ASK_EMBEDDING_DIM)
        context_window_size = read_int_parameter(ASK_CONTEXT_WINDOW_SIZE)
        negative_sample_count = read_int_parameter(ASK_NEGATIVE_SAMPLE_COUNT)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name="target_vocab_size",
            value=target_vocab_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name="context_vocab_size",
            value=context_vocab_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name="embedding_dim",
            value=embedding_dim
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name="context_window_size",
            value=context_window_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name="negative_sample_count",
            value=negative_sample_count
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="skipgram_sgns",
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_positive_sgns_parameter_again(
        self,
        parameter_fact,
        layer_index,
        parameter_name,
        value
    ):
        print()
        print("Invalid SGNS numeric value. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter("Enter a corrected positive value: ")

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name=parameter_name,
            value=new_value
        ))
