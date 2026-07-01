from experta import Rule, MATCH, AS

from src.facts import MenuChoice, SelectedLayer, RollbackRequest
from src.prompts import MAIN_MENU, ASK_ROLLBACK_LAYER


class RollbackRules:
    @Rule(
        AS.rollback_fact << RollbackRequest(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name,
            reason=MATCH.reason,
            message=MATCH.message
        ),
        AS.selected_layer_fact << SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name=MATCH.layer_name
        ),
        AS.menu_choice_fact << MenuChoice(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index
        ),
        salience=95
    )
    def rollback_layer_selection(
        self,
        rollback_fact,
        selected_layer_fact,
        menu_choice_fact,
        layer_index,
        previous_layer_index,
        layer_name,
        reason,
        message
    ):
        print()
        print("Rollback reason:")
        print(message)
        print()
        print("The selected layer will be removed from the current position.")
        print("Choose another layer for the same layer index.")
        print()

        self.retract(selected_layer_fact)
        self.retract(menu_choice_fact)
        self.retract(rollback_fact)

        print(MAIN_MENU)

        choice = input(ASK_ROLLBACK_LAYER)

        self.declare(MenuChoice(
            layer_index=layer_index,
            previous_layer_index=previous_layer_index,
            choice=choice
        ))
