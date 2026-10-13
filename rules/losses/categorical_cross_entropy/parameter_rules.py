from experta import AS, MATCH, NOT, Rule, TEST

from facts import OptionSelection, SelectedLayer
from prompts import (
    ASK_CCE_LABEL_FORMAT,
    ASK_LOSS_REDUCTION,
    CATEGORICAL_CROSS_ENTROPY_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class CategoricalCrossEntropyParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, layer_name="categorical_cross_entropy"),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="categorical_cross_entropy",
                question_key="label_format",
            ),
        ),
        salience=76,
    )
    def ask_cce_label_format(self, layer_index):
        print(CATEGORICAL_CROSS_ENTROPY_SELECTED)
        choice = read_text_parameter(ASK_CCE_LABEL_FORMAT)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="label_format",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=75
    )
    def resolve_cce_one_hot(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="label_format",
                choice="1",
                value="one_hot",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=75
    )
    def resolve_cce_class_index(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="label_format",
                choice="2",
                value="class_index",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=74
    )
    def ask_cce_label_format_again(self, option_fact, layer_index):
        print("Invalid label format. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_CCE_LABEL_FORMAT)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="label_format",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, layer_name="categorical_cross_entropy"),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="label_format",
            status="resolved",
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="categorical_cross_entropy",
                question_key="loss_reduction",
            ),
        ),
        salience=73
    )
    def ask_cce_reduction(self, layer_index):
        choice = read_text_parameter(ASK_LOSS_REDUCTION)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="loss_reduction",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=72
    )
    def resolve_cce_mean(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="loss_reduction",
                choice="1",
                value="mean",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=72
    )
    def resolve_cce_sum(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="loss_reduction",
                choice="2",
                value="sum",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="categorical_cross_entropy",
            question_key="loss_reduction",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=71
    )
    def ask_cce_reduction_again(self, option_fact, layer_index):
        print("Invalid reduction choice. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_LOSS_REDUCTION)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="categorical_cross_entropy",
                question_key="loss_reduction",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )
