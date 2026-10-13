from experta import Rule, NOT
from experta.fact import InitialFact

from facts import MenuChoice, OutputStatus
from prompts import MAIN_MENU, CHOOSE_FIRST_LAYER


from utils.console_control import read_int_parameter, read_text_parameter
class StartRules:
    @Rule(
        InitialFact(),
        NOT(OutputStatus(
            layer_index=0,
            output_name="main_menu",
            status="printed"
        )),
        salience=100
    )
    def show_main_menu(self):
        print(MAIN_MENU)

        choice = read_text_parameter(CHOOSE_FIRST_LAYER)

        self.declare(MenuChoice(
            layer_index=1,
            previous_layer_index=0,
            choice=choice
        ))

        self.declare(OutputStatus(
            layer_index=0,
            output_name="main_menu",
            status="printed"
        ))
