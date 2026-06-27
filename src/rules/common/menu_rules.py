from experta import Rule, MATCH, TEST, NOT

from src.facts import (
    MenuChoice,
    SelectedLayer,
    UnsupportedLayer,
    SessionStatus
)

from src.output_messages import UNSUPPORTED_LAYER_MESSAGE


class MenuRules:
    @Rule(
        MenuChoice(choice="1"),
        NOT(SelectedLayer(layer_index=1)),
        salience=90
    )
    def select_embedding_layer(self):
        self.declare(SelectedLayer(
            layer_index=1,
            layer_name="embedding",
            layer_title="Embedding Layer",
            source_name="source_transformers"
        ))

    @Rule(
        MenuChoice(choice=MATCH.choice),
        TEST(lambda choice: choice != "1"),
        NOT(UnsupportedLayer()),
        salience=80
    )
    def selected_layer_not_ready(self, choice):
        print(UNSUPPORTED_LAYER_MESSAGE)

        self.declare(UnsupportedLayer(
            choice=choice,
            reason="layer_rules_not_implemented_yet"
        ))

        self.declare(SessionStatus(
            status="ended",
            reason="unsupported_layer"
        ))
