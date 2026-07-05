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

        target_vocab_size = int(input(ASK_TARGET_VOCAB_SIZE))
        context_vocab_size = int(input(ASK_CONTEXT_VOCAB_SIZE))
        embedding_dim = int(input(ASK_EMBEDDING_DIM))
        context_window_size = int(input(ASK_CONTEXT_WINDOW_SIZE))
        negative_sample_count = int(input(ASK_NEGATIVE_SAMPLE_COUNT))

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

        new_value = int(input("Enter a corrected positive value: "))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="skipgram_sgns",
            parameter_name=parameter_name,
            value=new_value
        ))
