from experta import AS, MATCH, Rule, TEST

from facts import OptionSelection
from prompts import ASK_ATTENTION_SCORING_TYPE


from utils.console_control import read_int_parameter, read_text_parameter
class RNNAttentionOptionResolutionRules:
    @Rule(
        AS.option_fact << OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice="1", value="unknown", status="raw"),
        salience=76
    )
    def resolve_attention_dot_product(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice="1", value="dot_product", status="resolved"))

    @Rule(
        AS.option_fact << OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice="2", value="unknown", status="raw"),
        salience=76
    )
    def resolve_attention_bilinear(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice="2", value="bilinear", status="resolved"))

    @Rule(
        AS.option_fact << OptionSelection(layer_index=MATCH.layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice=MATCH.choice, value="unknown", status="raw"),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=75
    )
    def ask_attention_scoring_again(self, option_fact, layer_index):
        print("Invalid attention scoring type. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = read_text_parameter(ASK_ATTENTION_SCORING_TYPE)
        self.declare(OptionSelection(layer_index=layer_index, layer_name="rnn_attention", question_key="attention_scoring_type", choice=choice, value="unknown", status="raw"))
