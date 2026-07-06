from experta import Rule, MATCH, TEST, NOT, AS

from src.facts import OutputStatus, ContinueChoice, MenuChoice, SessionStatus, LayerMetadata
from src.prompts import MAIN_MENU, CONTINUE_MENU, CHOOSE_CONTINUE, CHOOSE_NEXT_LAYER


from src.utils.console_control import read_int_parameter, read_text_parameter
class ContinueRules:

    @Rule(
        OutputStatus(
            layer_index=MATCH.completed_layer_index,
            output_name="layer_result_printed",
            status="printed"
        ),
        LayerMetadata(
            layer_index=MATCH.completed_layer_index,
            terminal_status="terminal"
        ),
        NOT(SessionStatus(status="ended")),
        salience=26
    )
    def end_session_after_terminal_layer(self, completed_layer_index):
        self.declare(SessionStatus(
            status="ended",
            reason="terminal_layer_reached"
        ))

    @Rule(
        OutputStatus(
            layer_index=MATCH.completed_layer_index,
            output_name="layer_result_printed",
            status="printed"
        ),
        NOT(ContinueChoice(after_layer_index=MATCH.completed_layer_index)),
        NOT(SessionStatus(status="ended")),
        salience=25
    )
    def ask_continue_after_layer(self, completed_layer_index):
        print(CONTINUE_MENU)

        choice = read_text_parameter(CHOOSE_CONTINUE)

        self.declare(ContinueChoice(
            after_layer_index=completed_layer_index,
            choice=choice
        ))

    @Rule(
        ContinueChoice(
            after_layer_index=MATCH.completed_layer_index,
            choice="1"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.completed_layer_index,
            output_name="next_menu_displayed",
            status="printed"
        )),
        NOT(SessionStatus(status="ended")),
        salience=24
    )
    def show_next_layer_menu(self, completed_layer_index):
        next_layer_index = completed_layer_index + 1
        previous_layer_index = completed_layer_index

        print(MAIN_MENU)

        choice = read_text_parameter(CHOOSE_NEXT_LAYER)

        self.declare(MenuChoice(
            layer_index=next_layer_index,
            previous_layer_index=previous_layer_index,
            choice=choice
        ))

        self.declare(OutputStatus(
            layer_index=completed_layer_index,
            output_name="next_menu_displayed",
            status="printed"
        ))

    @Rule(
        ContinueChoice(
            after_layer_index=MATCH.completed_layer_index,
            choice="2"
        ),
        NOT(SessionStatus(status="ended")),
        salience=20
    )
    def end_session_by_user(self, completed_layer_index):
        self.declare(SessionStatus(
            status="ended",
            reason="user_finished"
        ))

    @Rule(
        AS.continue_choice_fact << ContinueChoice(
            after_layer_index=MATCH.completed_layer_index,
            choice=MATCH.choice
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        NOT(SessionStatus(status="ended")),
        salience=23
    )
    def ask_continue_choice_again(
        self,
        continue_choice_fact,
        completed_layer_index,
        choice
    ):
        print()
        print("Invalid choice. Please choose 1 or 2.")
        print(CONTINUE_MENU)

        self.retract(continue_choice_fact)

        new_choice = read_text_parameter(CHOOSE_CONTINUE)

        self.declare(ContinueChoice(
            after_layer_index=completed_layer_index,
            choice=new_choice
        ))
