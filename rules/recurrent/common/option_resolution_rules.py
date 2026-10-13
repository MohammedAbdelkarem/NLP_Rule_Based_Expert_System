from experta import AS, MATCH, Rule, TEST

from facts import OptionSelection
from prompts import ASK_RECURRENT_RETURN_MODE, ASK_RECURRENT_UNIT_TYPE


from utils.console_control import read_int_parameter, read_text_parameter
class RecurrentOptionResolutionRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_return_mode",
            choice="1",
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "simple_rnn",
            "lstm",
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        salience=76
    )
    def resolve_recurrent_return_full_sequence(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_return_mode",
            choice="1",
            value="full_sequence",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_return_mode",
            choice="2",
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "simple_rnn",
            "lstm",
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        salience=76
    )
    def resolve_recurrent_return_last_hidden(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_return_mode",
            choice="2",
            value="last_hidden",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_return_mode",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "simple_rnn",
            "lstm",
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=75
    )
    def ask_recurrent_return_mode_again(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        print("Invalid return mode. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_RECURRENT_RETURN_MODE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_return_mode",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_unit_type",
            choice="1",
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        salience=74
    )
    def resolve_recurrent_unit_simple_rnn(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_unit_type",
            choice="1",
            value="simple_rnn",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_unit_type",
            choice="2",
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        salience=74
    )
    def resolve_recurrent_unit_lstm(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_unit_type",
            choice="2",
            value="lstm",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key="recurrent_unit_type",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda layer_name: layer_name in (
            "stacked_rnn",
            "bidirectional_rnn",
        )),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_recurrent_unit_type_again(
        self,
        option_fact,
        layer_index,
        layer_name
    ):
        print("Invalid recurrent unit type. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_RECURRENT_UNIT_TYPE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name=layer_name,
            question_key="recurrent_unit_type",
            choice=choice,
            value="unknown",
            status="raw"
        ))
