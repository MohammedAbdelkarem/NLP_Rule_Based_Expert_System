from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter
from src.prompts import COSINE_SIMILARITY_SELECTED, ASK_VECTOR_DIM


from src.utils.console_control import read_int_parameter, read_text_parameter
class CosineSimilarityParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            parameter_name="vector_dim"
        )),
        salience=70
    )
    def ask_cosine_similarity_parameters(self, layer_index):
        print(COSINE_SIMILARITY_SELECTED)

        vector_dim = read_int_parameter(ASK_VECTOR_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="cosine_similarity",
            parameter_name="vector_dim",
            value=vector_dim
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="cosine_similarity",
            parameter_name="vector_dim",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_cosine_vector_dim_again(self, parameter_fact, layer_index, value):
        print()
        print("Invalid vector dimension. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_VECTOR_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="cosine_similarity",
            parameter_name="vector_dim",
            value=new_value
        ))
