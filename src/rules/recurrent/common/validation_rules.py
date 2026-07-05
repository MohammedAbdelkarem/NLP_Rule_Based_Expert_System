from experta import AS, MATCH, Rule, TEST

from src.facts import LayerParameter
from src.prompts import ASK_POSITIVE_INTEGER_VALUE


class RecurrentParameterValidationRules:
    @Rule(
        AS.parameter_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        TEST(lambda layer_name: layer_name in (
            "simple_rnn",
            "lstm",
            "stacked_rnn",
            "bidirectional_rnn",
            "rnn_encoder_decoder",
            "rnn_attention",
        )),
        TEST(lambda value: int(value) <= 0),
        salience=79
    )
    def ask_recurrent_positive_integer_again(
        self,
        parameter_fact,
        layer_index,
        layer_name,
        parameter_name
    ):
        print()
        print(f"Invalid value for {parameter_name}. It must be greater than 0.")
        self.retract(parameter_fact)
        new_value = int(input(ASK_POSITIVE_INTEGER_VALUE))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name=layer_name,
            parameter_name=parameter_name,
            value=new_value
        ))
