from experta import AS, MATCH, NOT, Rule, TEST

from src.facts import LayerParameter, OptionSelection, SelectedLayer
from src.prompts import (
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_SIGMOID_INPUT_KIND,
    ASK_SIGMOID_INPUT_RANK,
    SIGMOID_SELECTED,
)


class SigmoidParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="sigmoid",
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="sigmoid",
                question_key="sigmoid_input_data_kind",
            ),
        ),
        salience=75,
    )
    def ask_first_sigmoid_input_kind(self, layer_index):
        print(SIGMOID_SELECTED)
        choice = input(ASK_SIGMOID_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_sigmoid_input_kind_logits(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            choice="1",
            value="binary_logits",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        salience=73,
    )
    def rollback_invalid_sigmoid_input_kind(self, option_fact, layer_index, choice):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            choice=choice,
            value="invalid_for_sigmoid",
            status="resolved",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="sigmoid",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_data_kind",
            value="binary_logits",
            status="resolved",
        ),
        NOT(
            OptionSelection(
                layer_index=MATCH.layer_index,
                layer_name="sigmoid",
                question_key="sigmoid_input_rank",
            ),
        ),
        salience=72,
    )
    def ask_first_sigmoid_input_rank(self, layer_index):
        choice = input(ASK_SIGMOID_INPUT_RANK)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=71,
    )
    def resolve_sigmoid_rank_2(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice="1",
            value="rank_2",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=71,
    )
    def resolve_sigmoid_rank_3(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice="2",
            value="rank_3",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=70,
    )
    def ask_sigmoid_rank_again(self, option_fact, layer_index):
        print("Invalid Sigmoid input rank. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = input(ASK_SIGMOID_INPUT_RANK)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="sigmoid",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            value=MATCH.rank_value,
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="sigmoid",
                parameter_name="batch_size",
            ),
        ),
        salience=69,
    )
    def ask_sigmoid_batch_size(self, layer_index, rank_value):
        value = int(input(ASK_BATCH_SIZE))
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="sigmoid",
                parameter_name="batch_size",
                value=value,
            ),
        )

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="sigmoid",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="sigmoid",
            question_key="sigmoid_input_rank",
            value="rank_3",
            status="resolved",
        ),
        NOT(
            LayerParameter(
                layer_index=MATCH.layer_index,
                layer_name="sigmoid",
                parameter_name="sequence_length",
            ),
        ),
        salience=68,
    )
    def ask_sigmoid_sequence_length(self, layer_index):
        value = int(input(ASK_SEQUENCE_LENGTH))
        self.declare(
            LayerParameter(
                layer_index=layer_index,
                layer_name="sigmoid",
                parameter_name="sequence_length",
                value=value,
            ),
        )
