from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import SelectedLayer, LayerParameter, OptionSelection
from src.prompts import (
    FRAMING_WINDOWING_SELECTED,
    ASK_FRAME_LENGTH_MS,
    ASK_HOP_LENGTH_MS,
    ASK_WINDOW_FUNCTION,
)


class FramingWindowingParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            parameter_name="frame_length_ms"
        )),
        salience=70
    )
    def ask_framing_windowing_parameters(self, layer_index):
        print(FRAMING_WINDOWING_SELECTED)

        frame_length_ms = int(input(ASK_FRAME_LENGTH_MS))
        hop_length_ms = int(input(ASK_HOP_LENGTH_MS))
        window_choice = input(ASK_WINDOW_FUNCTION)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="framing_windowing",
            parameter_name="frame_length_ms",
            value=frame_length_ms
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="framing_windowing",
            parameter_name="hop_length_ms",
            value=hop_length_ms
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice=window_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_hann_window(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="1",
            value="hann",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_hamming_window(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="2",
            value="hamming",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="3",
            value="unknown",
            status="raw"
        ),
        salience=69
    )
    def resolve_rectangular_window(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice="3",
            value="rectangular",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=68
    )
    def ask_window_function_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid window function choice. Please choose 1, 2, or 3.")

        self.retract(option_fact)

        new_choice = input(ASK_WINDOW_FUNCTION)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="framing_windowing",
            question_key="window_function",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
