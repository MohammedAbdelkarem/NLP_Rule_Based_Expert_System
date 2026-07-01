from pathlib import Path

from experta import Rule, MATCH, NOT
from experta.fact import InitialFact

from src.facts import (
    Layer,
    ComponentGroup,
    LayerMetadata,
    WeightShape,
    OutputStatus,
    SessionStatus
)


class SummaryRules:
    @Rule(
        InitialFact(),
        NOT(OutputStatus(
            layer_index=0,
            output_name="session_summary",
            status="initialized"
        )),
        salience=109
    )
    def initialize_session_summary_file(self):
        Path("outputs").mkdir(exist_ok=True)

        Path("outputs/session_summary.txt").write_text(
            "Session Summary\n"
            "===============\n\n"
        )

        self.declare(OutputStatus(
            layer_index=0,
            output_name="session_summary",
            status="initialized"
        ))

    @Rule(
        ComponentGroup(
            group_index=MATCH.group_index,
            group_name=MATCH.group_name,
            group_title=MATCH.group_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        LayerMetadata(
            layer_index=MATCH.group_index,
            layer_name=MATCH.group_name,
            layer_kind=MATCH.layer_kind,
            layer_role=MATCH.layer_role,
            repeat_policy=MATCH.repeat_policy,
            consecutive_repeat_policy=MATCH.consecutive_repeat_policy,
            terminal_status=MATCH.terminal_status
        ),
        WeightShape(
            layer_index=MATCH.group_index,
            layer_name=MATCH.group_name,
            weight_name=MATCH.weight_name,
            shape=MATCH.weight_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.group_index,
            output_name="component_group_summary_written",
            status="written"
        )),
        salience=46
    )
    def write_component_group_summary(
        self,
        group_index,
        group_name,
        group_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape,
        layer_kind,
        layer_role,
        repeat_policy,
        consecutive_repeat_policy,
        terminal_status,
        weight_name,
        weight_shape
    ):
        summary_text = (
            f"Layer {group_index}: {group_title}\n"
            f"Layer name: {group_name}\n"
            f"Layer kind: {layer_kind}\n"
            f"Layer role: {layer_role}\n"
            f"Repeat policy: {repeat_policy}\n"
            f"Consecutive repeat policy: {consecutive_repeat_policy}\n"
            f"Terminal status: {terminal_status}\n"
            f"Internal layers: {internal_layers}\n"
            f"Combine operation: {combine_operation}\n"
            f"Input shape: {input_shape}\n"
            f"Parameters: {weight_name} = {weight_shape}\n"
            f"Output shape: {output_shape}\n"
            "-----------------------------\n\n"
        )

        with Path("outputs/session_summary.txt").open("a") as file:
            file.write(summary_text)

        self.declare(OutputStatus(
            layer_index=group_index,
            output_name="component_group_summary_written",
            status="written"
        ))

        self.declare(OutputStatus(
            layer_index=group_index,
            output_name="layer_summary_written",
            status="written"
        ))

    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        LayerMetadata(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            layer_kind=MATCH.layer_kind,
            layer_role=MATCH.layer_role,
            repeat_policy=MATCH.repeat_policy,
            consecutive_repeat_policy=MATCH.consecutive_repeat_policy,
            terminal_status=MATCH.terminal_status
        ),
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            weight_name=MATCH.weight_name,
            shape=MATCH.weight_shape
        ),
        NOT(ComponentGroup(group_index=MATCH.layer_index)),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="layer_summary_written",
            status="written"
        )),
        salience=45
    )
    def write_layer_summary(
        self,
        layer_index,
        layer_name,
        layer_title,
        input_shape,
        output_shape,
        layer_kind,
        layer_role,
        repeat_policy,
        consecutive_repeat_policy,
        terminal_status,
        weight_name,
        weight_shape
    ):
        summary_text = (
            f"Layer {layer_index}: {layer_title}\n"
            f"Layer name: {layer_name}\n"
            f"Layer kind: {layer_kind}\n"
            f"Layer role: {layer_role}\n"
            f"Repeat policy: {repeat_policy}\n"
            f"Consecutive repeat policy: {consecutive_repeat_policy}\n"
            f"Terminal status: {terminal_status}\n"
            f"Input shape: {input_shape}\n"
            f"Parameters: {weight_name} = {weight_shape}\n"
            f"Output shape: {output_shape}\n"
            "-----------------------------\n\n"
        )

        with Path("outputs/session_summary.txt").open("a") as file:
            file.write(summary_text)

        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name="layer_summary_written",
            status="written"
        ))

    @Rule(
        SessionStatus(
            status="ended",
            reason=MATCH.reason
        ),
        NOT(OutputStatus(
            layer_index=0,
            output_name="session_end_written",
            status="written"
        )),
        salience=5
    )
    def write_session_end(self, reason):
        end_text = (
            "Session ended\n"
            f"Reason: {reason}\n"
        )

        with Path("outputs/session_summary.txt").open("a") as file:
            file.write(end_text)

        self.declare(OutputStatus(
            layer_index=0,
            output_name="session_end_written",
            status="written"
        ))
