from experta import Rule, MATCH, TEST, NOT, AS

from src.facts import MenuChoice, SelectedLayer, SessionStatus
from src.prompts import MAIN_MENU, ASK_ROLLBACK_LAYER


class MenuRules:
    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="1"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_input_encoding(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="input_encoding",
            layer_title="Input Encoding Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="2"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_layer_norm(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="layer_norm",
            layer_title="LayerNorm Layer",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="3"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_multi_head_attention(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="multi_head_attention",
            layer_title="Multi-Head Attention Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="4"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_residual_connection(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="residual_connection",
            layer_title="Residual Connection Layer",
            source_name="transformers"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="5"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_feed_forward(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="feed_forward",
            layer_title="Feed-Forward Network Layer Group",
            source_name="transformers"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="6"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_relu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="relu",
            layer_title="ReLU Activation Layer",
            source_name="activations"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="7"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_gelu(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="gelu",
            layer_title="GELU Activation Layer",
            source_name="activations"
        ))


    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="8"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_position_wise_linear(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="position_wise_linear",
            layer_title="Position-wise Linear Layer",
            source_name="projections"
        ))



    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="9"
        ),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=90
    )
    def select_softmax(self, layer_index, previous_layer_index):
        self.declare(SelectedLayer(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            layer_name="softmax",
            layer_title="Softmax Layer",
            source_name="activations"
        ))

    @Rule(
        MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice="0"
        ),
        NOT(SessionStatus(status="ended")),
        salience=90
    )
    def end_session_from_layer_menu(self, layer_index, previous_layer_index):
        self.declare(SessionStatus(
            status="ended",
            reason="user_stopped_from_layer_menu"
        ))

    @Rule(
        AS.menu_choice_fact << MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            choice=MATCH.choice
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        TEST(lambda choice: choice != "4"),
        TEST(lambda choice: choice != "5"),
        TEST(lambda choice: choice != "6"),
        TEST(lambda choice: choice != "7"),
        TEST(lambda choice: choice != "8"),
        TEST(lambda choice: choice != "9"),
        TEST(lambda choice: choice != "0"),
        NOT(SelectedLayer(layer_index=MATCH.layer_index)),
        salience=89
    )
    def ask_layer_choice_again(
        self,
        menu_choice_fact,
        layer_index,
        previous_layer_index,
        choice
    ):
        print()
        print("Invalid layer choice. Please choose 1, 2, 3, 4, 5, 6, 7, 8, 9, or 0.")
        print(MAIN_MENU)

        self.retract(menu_choice_fact)

        new_choice = input(ASK_ROLLBACK_LAYER)

        self.declare(MenuChoice(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            choice=new_choice
        ))
