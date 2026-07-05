from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    EMBEDDING_BIAS_ANALYSIS_SELECTED,
    ASK_TARGET_WORD_COUNT,
    ASK_ATTRIBUTE_WORD_COUNT,
    ASK_EMBEDDING_DIM,
)


class EmbeddingBiasAnalysisParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="target_word_count"
        )),
        salience=70
    )
    def ask_embedding_bias_analysis_parameters(self, layer_index):
        print(EMBEDDING_BIAS_ANALYSIS_SELECTED)

        target_word_count = int(input(ASK_TARGET_WORD_COUNT))
        attribute_word_count = int(input(ASK_ATTRIBUTE_WORD_COUNT))
        embedding_dim = int(input(ASK_EMBEDDING_DIM))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="target_word_count",
            value=target_word_count
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="attribute_word_count",
            value=attribute_word_count
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name="embedding_dim",
            value=embedding_dim
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_positive_bias_analysis_parameter_again(
        self,
        parameter_fact,
        layer_index,
        parameter_name,
        value
    ):
        print()
        print("Invalid bias analysis numeric value. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = int(input("Enter a corrected positive value: "))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_bias_analysis",
            parameter_name=parameter_name,
            value=new_value
        ))
