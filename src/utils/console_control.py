import builtins

from src.facts import MenuChoice, SessionStatus
from src.prompts import MAIN_MENU, ASK_GLOBAL_ROLLBACK_LAYER


ORIGINAL_INPUT = builtins.input
COMMAND_FOOTER = """

[Always available: R = rollback current step | E = end session]
"""
EMPTY_CONTEXT = {
    "layer_index": 1,
    "previous_layer_index": 0,
}
CONTEXT_FACT_CLASSES = {
    "SelectedLayer": True,
    "MenuChoice": True,
}
NEGATIVE_LAYER_INDEX = -999999


class EndSessionSignal(Exception):
    pass


class RollbackSignal(Exception):
    pass


def keep_user_value(raw_value):
    return raw_value


def raise_rollback_signal(raw_value):
    raise RollbackSignal()


def raise_end_session_signal(raw_value):
    raise EndSessionSignal()


COMMAND_HANDLERS = {
    "r": raise_rollback_signal,
    "rollback": raise_rollback_signal,
    ":r": raise_rollback_signal,
    ":rollback": raise_rollback_signal,
    "e": raise_end_session_signal,
    "end": raise_end_session_signal,
    "exit": raise_end_session_signal,
    "quit": raise_end_session_signal,
    "q": raise_end_session_signal,
    ":e": raise_end_session_signal,
    ":end": raise_end_session_signal,
    ":exit": raise_end_session_signal,
    ":quit": raise_end_session_signal,
    ":q": raise_end_session_signal,
}


def normalize_command(raw_value):
    return raw_value.strip().lower()


def controlled_input(prompt=""):
    raw_value = ORIGINAL_INPUT(f"{prompt}{COMMAND_FOOTER}")
    command_key = normalize_command(raw_value)
    return COMMAND_HANDLERS.get(command_key, keep_user_value)(raw_value)


def read_text_parameter(prompt=""):
    return controlled_input(prompt)


def cast_integer_value(raw_value, prompt):
    try:
        return int(raw_value)
    except ValueError:
        print("Invalid integer value. Please enter a number, R, or E.")
        return read_int_parameter(prompt)


def read_int_parameter(prompt=""):
    raw_value = controlled_input(prompt)
    return cast_integer_value(raw_value, prompt)


def install_console_controls():
    builtins.input = controlled_input


def fact_class_name(item):
    return item[1].__class__.__name__


def is_context_item(item):
    return CONTEXT_FACT_CLASSES.get(fact_class_name(item), False)


def item_fact(item):
    return item[1]


def fact_layer_index(fact):
    return int(fact.get("layer_index", NEGATIVE_LAYER_INDEX))


def latest_interaction_context(engine):
    context_items = filter(is_context_item, tuple(engine.facts.items()))
    context_facts = tuple(map(item_fact, context_items)) + (EMPTY_CONTEXT,)
    return max(context_facts, key=fact_layer_index)


def fact_target_index(fact):
    return int(fact.get(
        "layer_index",
        fact.get(
            "group_index",
            fact.get(
                "component_index",
                fact.get("after_layer_index", NEGATIVE_LAYER_INDEX),
            ),
        ),
    ))


def make_target_predicate(layer_index):
    return lambda item: fact_target_index(item[1]) >= layer_index


def make_retract_action(engine):
    return lambda item: engine.retract(item[0])


def retract_current_step(engine, layer_index):
    target_items = filter(make_target_predicate(layer_index), tuple(engine.facts.items()))
    tuple(map(make_retract_action(engine), target_items))


def declare_end_session(engine):
    engine.declare(SessionStatus(
        status="ended",
        reason="user_global_end_command",
    ))
    print()
    print("Session ended by user command.")


def ask_replacement_layer(engine, layer_index, previous_layer_index):
    print()
    print("Rollback command received.")
    print("The current step facts were removed.")
    print("Choose a replacement layer for the same position.")
    print()
    print(MAIN_MENU)
    choice = read_text_parameter(ASK_GLOBAL_ROLLBACK_LAYER)
    engine.declare(MenuChoice(
        layer_index=layer_index,
        previous_layer_index=previous_layer_index,
        choice=choice,
    ))


def handle_global_rollback(engine):
    context = latest_interaction_context(engine)
    layer_index = int(context.get("layer_index", 1))
    previous_layer_index = int(context.get("previous_layer_index", layer_index - 1))
    retract_current_step(engine, layer_index)
    ask_replacement_layer(engine, layer_index, previous_layer_index)


def run_engine_action(engine):
    engine.run()


def rollback_then_run_action(engine):
    handle_global_rollback(engine)
    engine.run()


def execute_with_global_controls(action, engine):
    try:
        action(engine)
    except EndSessionSignal:
        declare_end_session(engine)
    except RollbackSignal:
        execute_with_global_controls(rollback_then_run_action, engine)


def run_controlled_engine(engine):
    execute_with_global_controls(run_engine_action, engine)
