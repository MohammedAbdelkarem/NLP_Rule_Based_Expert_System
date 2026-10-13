from experta import AS, MATCH, NOT, Rule, TEST

from facts import LayerParameter, OptionSelection, SelectedLayer
from prompts import (
    ASK_BATCH_SIZE,
    ASK_INPUT_DIM,
    ASK_MULTICLASS_CLASS_COUNT,
    ASK_MULTICLASS_CLASSIFIER_INPUT_RANK,
    ASK_SEQUENCE_LENGTH,
    MULTINOMIAL_LOGISTIC_REGRESSION_HEAD_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class MultinomialLogisticRegressionHeadParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multinomial_logistic_regression_head",
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="multinomial_logistic_regression_head",
                question_key="classifier_input_rank",
            ),
        ),
        salience=75,
    )
    def ask_first_multinomial_head_input_rank(self, layer_index):
        print(MULTINOMIAL_LOGISTIC_REGRESSION_HEAD_SELECTED)
        choice = read_text_parameter(ASK_MULTICLASS_CLASSIFIER_INPUT_RANK)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                question_key="classifier_input_rank",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_multinomial_head_rank_2(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                question_key="classifier_input_rank",
                choice="1",
                value="rank_2",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74
    )
    def resolve_multinomial_head_rank_3(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                question_key="classifier_input_rank",
                choice="2",
                value="rank_3",
                status="resolved",
            ),
        )

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"), TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_multinomial_head_rank_again(self, option_fact, layer_index):
        print("Invalid classifier input rank. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_MULTICLASS_CLASSIFIER_INPUT_RANK)
        self.declare(
            OptionSelection(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                question_key="classifier_input_rank",
                choice=choice,
                value="unknown",
                status="raw",
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multinomial_logistic_regression_head",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            value=MATCH.rank_value,
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="batch_size",
            ),
        ),
        salience=72
    )
    def ask_multinomial_head_batch_size(self, layer_index, rank_value):
        value = read_int_parameter(ASK_BATCH_SIZE)
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="batch_size",
                value=value,
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multinomial_logistic_regression_head",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            value="rank_3",
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="sequence_length",
            ),
        ),
        salience=71
    )
    def ask_multinomial_head_sequence_length(self, layer_index):
        value = read_int_parameter(ASK_SEQUENCE_LENGTH)
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="sequence_length",
                value=value,
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multinomial_logistic_regression_head",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
            question_key="classifier_input_rank",
            value=MATCH.rank_value,
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="input_dim",
            ),
        ),
        salience=70
    )
    def ask_multinomial_head_input_dim(self, layer_index, rank_value):
        value = read_int_parameter(ASK_INPUT_DIM)
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="input_dim",
                value=value,
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="multinomial_logistic_regression_head",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="num_classes",
            ),
        ),
        salience=69
    )
    def ask_multinomial_head_num_classes(self, layer_index):
        value = read_int_parameter(ASK_MULTICLASS_CLASS_COUNT)
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="multinomial_logistic_regression_head",
                parameter_name="num_classes",
                value=value,
            ),
        )
