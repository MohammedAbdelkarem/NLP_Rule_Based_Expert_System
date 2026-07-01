from pathlib import Path

from experta import Rule, MATCH, NOT
from experta.fact import InitialFact

from src.facts import CalculationStep, OutputStatus


class CalculationTraceRules:
    def write_step_to_file(
        self,
        layer_index,
        layer_name,
        step_key,
        step_order,
        title,
        formula,
        substitution,
        result
    ):
        text = (
            f"Layer {layer_index}: {layer_name}\n"
            f"Step {step_order}: {title}\n"
            f"Formula:\n{formula}\n"
            f"Substitution:\n{substitution}\n"
            f"Result:\n{result}\n"
            "-----------------------------\n\n"
        )

        with Path("outputs/calculation_trace.txt").open("a") as file:
            file.write(text)

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name=step_key,
            status="written"
        ))

    @Rule(
        InitialFact(),
        NOT(OutputStatus(
            layer_index=0,
            output_name="calculation_trace",
            status="initialized"
        )),
        salience=110
    )
    def initialize_calculation_trace_file(self):
        Path("outputs").mkdir(exist_ok=True)

        Path("outputs/calculation_trace.txt").write_text(
            "Calculation Trace\n"
            "=================\n\n"
        )

        self.declare(OutputStatus(
            layer_index=0,
            output_name="calculation_trace",
            status="initialized"
        ))

    @Rule(
        CalculationStep(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            step_key=MATCH.step_key,
            previous_step_key="none",
            step_order=MATCH.step_order,
            title=MATCH.title,
            formula=MATCH.formula,
            substitution=MATCH.substitution,
            result=MATCH.result
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.step_key,
            status="written"
        )),
        salience=40
    )
    def write_first_calculation_step(
        self,
        layer_index,
        layer_name,
        step_key,
        step_order,
        title,
        formula,
        substitution,
        result
    ):
        self.write_step_to_file(
            layer_index,
            layer_name,
            step_key,
            step_order,
            title,
            formula,
            substitution,
            result
        )

    @Rule(
        CalculationStep(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            step_key=MATCH.step_key,
            previous_step_key=MATCH.previous_step_key,
            step_order=MATCH.step_order,
            title=MATCH.title,
            formula=MATCH.formula,
            substitution=MATCH.substitution,
            result=MATCH.result
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.previous_step_key,
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.step_key,
            status="written"
        )),
        salience=30
    )
    def write_next_calculation_step(
        self,
        layer_index,
        layer_name,
        step_key,
        previous_step_key,
        step_order,
        title,
        formula,
        substitution,
        result
    ):
        self.write_step_to_file(
            layer_index,
            layer_name,
            step_key,
            step_order,
            title,
            formula,
            substitution,
            result
        )
