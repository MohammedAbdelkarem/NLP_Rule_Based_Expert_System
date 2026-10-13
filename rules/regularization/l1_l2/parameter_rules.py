from experta import AS, MATCH, NOT, Rule, TEST

from facts import LayerParameter, OptionSelection, SelectedLayer
from prompts import (
    ASK_REGULARIZATION_LAMBDA_EXPONENT,
    ASK_REGULARIZATION_TYPE,
    REGULARIZATION_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class L1L2RegularizationParameterRules:
    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, layer_name="l1_l2_regularization"),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="l1_l2_regularization",
                question_key="regularization_type",
            ),
        ),
        salience=75,
    )
    def ask_regularization_type(self, layer_index):
        print(REGULARIZATION_SELECTED)
        choice = read_text_parameter(ASK_REGULARIZATION_TYPE)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="l1_l2_regularization",
                question_key="regularization_type",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="l1_l2_regularization",
            question_key="regularization_type",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_l1_regularization(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="l1_l2_regularization",
                question_key="regularization_type",
                choice="1",
                value="l1",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="l1_l2_regularization",
            question_key="regularization_type",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_l2_regularization(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="l1_l2_regularization",
                question_key="regularization_type",
                choice="2",
                value="l2",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="l1_l2_regularization",
            question_key="regularization_type",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_regularization_type_again(self, option_fact, layer_index):
        print("Invalid regularization choice. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_REGULARIZATION_TYPE)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="l1_l2_regularization",
                question_key="regularization_type",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        SelectedLayer(layer_index=MATCH.layer_index, layer_name="l1_l2_regularization"),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="l1_l2_regularization",
            question_key="regularization_type",
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="l1_l2_regularization",
                parameter_name="lambda_exponent",
            ),
        ),
        salience=72
    )
    def ask_regularization_lambda_exponent(self, layer_index):
        value = read_int_parameter(ASK_REGULARIZATION_LAMBDA_EXPONENT)
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="l1_l2_regularization",
                parameter_name="lambda_exponent",
                value=value,
            ),
        )
