from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    COUNT_BASED_EMBEDDING_SELECTED,
    ASK_WORD_VOCAB_SIZE,
    ASK_CONTEXT_VOCAB_SIZE,
    ASK_CONTEXT_WINDOW_SIZE,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class CountBasedEmbeddingParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="vocab_size"
        )),
        salience=70
    )
    def ask_count_based_embedding_parameters(self, layer_index):
        print(COUNT_BASED_EMBEDDING_SELECTED)

        vocab_size = read_int_parameter(ASK_WORD_VOCAB_SIZE)
        context_vocab_size = read_int_parameter(ASK_CONTEXT_VOCAB_SIZE)
        context_window_size = read_int_parameter(ASK_CONTEXT_WINDOW_SIZE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="vocab_size",
            value=vocab_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_vocab_size",
            value=context_vocab_size
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_window_size",
            value=context_window_size
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="vocab_size",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_count_based_vocab_size_again(self, parameter_fact, layer_index, value):
        print()
        print("Invalid word vocabulary size. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_WORD_VOCAB_SIZE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="vocab_size",
            value=new_value
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_vocab_size",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_count_based_context_vocab_size_again(
        self,
        parameter_fact,
        layer_index,
        value
    ):
        print()
        print("Invalid context vocabulary size. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_CONTEXT_VOCAB_SIZE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_vocab_size",
            value=new_value
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_window_size",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_count_based_context_window_size_again(
        self,
        parameter_fact,
        layer_index,
        value
    ):
        print()
        print("Invalid context window size. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_CONTEXT_WINDOW_SIZE)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="count_based_embedding",
            parameter_name="context_window_size",
            value=new_value
        ))
