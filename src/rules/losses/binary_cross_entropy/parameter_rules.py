from experta import AS, MATCH, NOT, Rule, TEST

from src.facts import OptionSelection, SelectedLayer
from src.prompts import ASK_LOSS_REDUCTION, BINARY_CROSS_ENTROPY_SELECTED


from src.utils.console_control import read_int_parameter, read_text_parameter
class BinaryCrossEntropyParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, layer_name="binary_cross_entropy"),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="binary_cross_entropy",
                question_key="loss_reduction",
            ),
        ),
        salience=75,
    )
    def ask_binary_cross_entropy_reduction(self, layer_index):
        print(BINARY_CROSS_ENTROPY_SELECTED)
        choice = read_text_parameter(ASK_LOSS_REDUCTION)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="binary_cross_entropy",
                question_key="loss_reduction",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            question_key="loss_reduction",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_bce_mean(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="binary_cross_entropy",
                question_key="loss_reduction",
                choice="1",
                value="mean",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            question_key="loss_reduction",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_bce_sum(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="binary_cross_entropy",
                question_key="loss_reduction",
                choice="2",
                value="sum",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="binary_cross_entropy",
            question_key="loss_reduction",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_bce_reduction_again(self, option_fact, layer_index):
        print("Invalid reduction choice. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_LOSS_REDUCTION)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="binary_cross_entropy",
                question_key="loss_reduction",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )
