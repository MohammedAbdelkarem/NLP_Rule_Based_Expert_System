from experta import Rule, MATCH, NOT, AS, TEST

from facts import SelectedLayer, LayerParameter
from prompts import (
    EMBEDDING_VISUALIZATION_SELECTED,
    ASK_NUM_EMBEDDED_ITEMS,
    ASK_EMBEDDING_DIM,
    ASK_VISUALIZATION_DIM,
)


from utils.console_control import read_int_parameter, read_text_parameter
class EmbeddingVisualizationParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="num_items"
        )),
        salience=70
    )
    def ask_embedding_visualization_parameters(self, layer_index):
        print(EMBEDDING_VISUALIZATION_SELECTED)

        num_items = read_int_parameter(ASK_NUM_EMBEDDED_ITEMS)
        embedding_dim = read_int_parameter(ASK_EMBEDDING_DIM)
        visualization_dim = read_int_parameter(ASK_VISUALIZATION_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name="num_items",
            value=num_items
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name="embedding_dim",
            value=embedding_dim
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=visualization_dim
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        TEST(lambda value: int(value) <= 0),
        salience=69
    )
    def ask_positive_visualization_parameter_again(
        self,
        parameter_fact,
        layer_index,
        parameter_name,
        value
    ):
        print()
        print("Invalid visualization numeric value. It must be greater than 0.")

        self.retract(parameter_fact)

        new_value = read_int_parameter("Enter a corrected positive value: ")

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name=parameter_name,
            value=new_value
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) < 2),
        salience=68
    )
    def ask_visualization_dim_again_below_two(
        self,
        parameter_fact,
        layer_index,
        value
    ):
        print()
        print("Invalid visualization dimension. Use 2 or 3.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_VISUALIZATION_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=new_value
        ))

    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=MATCH.value
        ),
        TEST(lambda value: int(value) > 3),
        salience=68
    )
    def ask_visualization_dim_again_above_three(
        self,
        parameter_fact,
        layer_index,
        value
    ):
        print()
        print("Invalid visualization dimension. Use 2 or 3.")

        self.retract(parameter_fact)

        new_value = read_int_parameter(ASK_VISUALIZATION_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_visualization",
            parameter_name="visualization_dim",
            value=new_value
        ))
