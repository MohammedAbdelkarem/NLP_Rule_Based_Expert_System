from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import (
    EMBEDDING_EVALUATION_SELECTED,
    ASK_EVALUATION_PAIR_COUNT,
    ASK_EMBEDDING_DIM,
    ASK_BENCHMARK_COUNT,
)


class EmbeddingEvaluationParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            parameter_name="evaluation_pair_count"
        )),
        salience=70
    )
    def ask_embedding_evaluation_parameters(self, layer_index):
        print(EMBEDDING_EVALUATION_SELECTED)

        evaluation_pair_count = int(input(ASK_EVALUATION_PAIR_COUNT))
        embedding_dim = int(input(ASK_EMBEDDING_DIM))
        benchmark_count = int(input(ASK_BENCHMARK_COUNT))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_evaluation",
            parameter_name="evaluation_pair_count",
            value=evaluation_pair_count
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_evaluation",
            parameter_name="embedding_dim",
            value=embedding_dim
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_evaluation",
            parameter_name="benchmark_count",
            value=benchmark_count
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_evaluation",
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_positive_evaluation_parameter_again(
        self,
        parameter_fact,
        layer_index,
        parameter_name,
        value
    ):
        print()
        print("Invalid evaluation numeric value. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = int(input("Enter a corrected positive value: "))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_evaluation",
            parameter_name=parameter_name,
            value=new_value
        ))
