from pathlib import Path

from experta import Rule, MATCH, NOT

from src.facts import (
    OutputStatus,
    ContinueChoice,
    SessionStatus,
    Layer,
    WeightShape
)

from src.prompts import CONTINUE_MENU, CHOOSE_CONTINUE
from src.output_messages import SUMMARY_SAVED_MESSAGE


class SummaryRules:
    @Rule(
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="embedding_result",
            status="printed"
        ),
        NOT(ContinueChoice()),
        salience=20
    )
    def ask_continue_question(self, layer_index):
        print(CONTINUE_MENU)
        choice = input(CHOOSE_CONTINUE)

        self.declare(ContinueChoice(choice=choice))

    @Rule(
        ContinueChoice(choice="2"),
        NOT(SessionStatus(status="ended")),
        salience=10
    )
    def end_session_by_user(self):
        self.declare(SessionStatus(
            status="ended",
            reason="user_finished"
        ))

    @Rule(
        ContinueChoice(choice="1"),
        NOT(SessionStatus(status="ended")),
        salience=10
    )
    def next_layer_not_ready(self):
        print("Next layer flow will be added after finishing the first layer correctly.")

        self.declare(SessionStatus(
            status="ended",
            reason="next_layer_not_ready"
        ))

    @Rule(
        SessionStatus(status="ended"),
        Layer(
            layer_index=MATCH.layer_index,
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            weight_name=MATCH.weight_name,
            shape=MATCH.weight_shape
        ),
        NOT(OutputStatus(
            layer_index=0,
            output_name="session_summary",
            status="saved"
        )),
        salience=0
    )
    def save_session_summary(
        self,
        layer_index,
        layer_title,
        input_shape,
        output_shape,
        weight_name,
        weight_shape
    ):
        summary_text = f"""
Session Summary

Layer {layer_index}: {layer_title}

Input shape:
{input_shape}

Weight:
{weight_name} = {weight_shape}

Output shape:
{output_shape}
"""

        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/session_summary.txt").write_text(summary_text)

        print(SUMMARY_SAVED_MESSAGE)

        self.declare(OutputStatus(
            layer_index=0,
            output_name="session_summary",
            status="saved"
        ))
