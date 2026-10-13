from pathlib import Path

from experta import Rule, MATCH, NOT, TEST
from experta.fact import InitialFact

from facts import (
    CalculationStep,
    ComponentGroup,
    IntermediateTensor,
    Layer,
    LayerParameter,
    OptionSelection,
    OutputStatus,
    TensorInfo,
    TrainingComponent,
    WeightShape,
)


class CalculationTraceRules:
    def append_trace_text(self, text):
        with Path("outputs/calculation_trace.txt").open("a") as file:
            file.write(text)

    def mark_written(self, layer_index, output_name, status):
        self.declare(OutputStatus(
            layer_index=layer_index,
            output_name=output_name,
            status=status
        ))

    def write_layer_header_text(
        self,
        layer_index,
        layer_name,
        layer_title,
        input_shape,
        output_shape
    ):
        self.append_trace_text(
            "============================================================\n"
            f"Layer {layer_index}: {layer_title}\n"
            "============================================================\n"
            "Component type: Neural network layer\n"
            f"Internal name: {layer_name}\n"
            f"Declared input shape: {input_shape}\n"
            f"Declared output shape: {output_shape}\n\n"
        )

        self.mark_written(layer_index, "calculation_header", "written")

    def write_component_group_header_text(
        self,
        layer_index,
        group_name,
        group_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape
    ):
        self.append_trace_text(
            "============================================================\n"
            f"Layer {layer_index}: {group_title}\n"
            "============================================================\n"
            "Component type: Component group\n"
            f"Internal name: {group_name}\n"
            f"Internal components: {internal_layers}\n"
            f"Combination / transformation operation: {combine_operation}\n"
            f"Declared input shape: {input_shape}\n"
            f"Declared output shape: {output_shape}\n\n"
        )

        self.mark_written(layer_index, "calculation_header", "written")

    def write_training_component_header_text(
        self,
        component_index,
        component_name,
        component_title,
        component_type,
        input_shape,
        output_shape,
        used_during
    ):
        self.append_trace_text(
            "============================================================\n"
            f"Layer {component_index}: {component_title}\n"
            "============================================================\n"
            "Component type: Training / optimization component\n"
            f"Internal name: {component_name}\n"
            f"Training component type: {component_type}\n"
            f"Declared input shape: {input_shape}\n"
            f"Declared output shape: {output_shape}\n"
            f"Used during: {used_during}\n\n"
        )

        self.mark_written(component_index, "calculation_header", "written")

    def write_layer_parameter_text(
        self,
        layer_index,
        layer_name,
        parameter_name,
        value
    ):
        self.append_trace_text(
            "Input / configuration parameter\n"
            f"Layer: {layer_name}\n"
            f"Parameter: {parameter_name}\n"
            f"Value: {value}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, parameter_name, "parameter_written")

    def write_option_selection_text(
        self,
        layer_index,
        layer_name,
        question_key,
        choice,
        value
    ):
        self.append_trace_text(
            "Input / configuration choice\n"
            f"Layer: {layer_name}\n"
            f"Question key: {question_key}\n"
            f"User choice: {choice}\n"
            f"Resolved value: {value}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, question_key, "option_written")

    def write_tensor_info_text(
        self,
        section_title,
        layer_index,
        layer_name,
        tensor_name,
        role,
        dims_count,
        shape,
        data_type,
        source
    ):
        self.append_trace_text(
            f"{section_title}\n"
            f"Layer: {layer_name}\n"
            f"Tensor name: {tensor_name}\n"
            f"Role: {role}\n"
            f"Dimensions count: {dims_count}\n"
            f"Shape: {shape}\n"
            f"Data type: {data_type}\n"
            f"Source: {source}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, tensor_name, "tensor_written")

    def write_intermediate_tensor_text(
        self,
        layer_index,
        layer_name,
        tensor_name,
        shape,
        data_type,
        source
    ):
        self.append_trace_text(
            "Intermediate calculation tensor\n"
            f"Layer: {layer_name}\n"
            f"Tensor name: {tensor_name}\n"
            f"Shape: {shape}\n"
            f"Data type: {data_type}\n"
            f"Source: {source}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, tensor_name, "intermediate_tensor_written")

    def write_weight_shape_text(
        self,
        layer_index,
        layer_name,
        weight_name,
        shape
    ):
        self.append_trace_text(
            "Learnable / fixed weight or parameter shape\n"
            f"Layer: {layer_name}\n"
            f"Weight / parameter name: {weight_name}\n"
            f"Shape: {shape}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, weight_name, "weight_written")

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
        self.append_trace_text(
            "Calculation step\n"
            f"Layer: {layer_name}\n"
            f"Step {step_order}: {title}\n"
            f"Formula:\n{formula}\n"
            f"Substitution:\n{substitution}\n"
            f"Result:\n{result}\n"
            "-----------------------------\n\n"
        )

        self.mark_written(layer_index, step_key, "written")

    @Rule(
        InitialFact(),
        NOT(OutputStatus(
            layer_index=0,
            output_name="calculation_trace",
            status="initialized"
        )),
        salience=120
    )
    def initialize_calculation_trace_file(self):
        Path("outputs").mkdir(exist_ok=True)

        Path("outputs/calculation_trace.txt").write_text(
            "Calculation Trace\n"
            "=================\n\n"
            "This file records the declared input information, configuration values,\n"
            "weight or fixed-parameter shapes, intermediate tensors, calculation formulas,\n"
            "and output tensors for each selected component.\n\n"
        )

        self.mark_written(0, "calculation_trace", "initialized")

    @Rule(
        Layer(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            layer_title=MATCH.layer_title,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        )),
        salience=100
    )
    def write_layer_header(
        self,
        layer_index,
        layer_name,
        layer_title,
        input_shape,
        output_shape
    ):
        self.write_layer_header_text(
            layer_index,
            layer_name,
            layer_title,
            input_shape,
            output_shape
        )

    @Rule(
        ComponentGroup(
            group_index=MATCH.layer_index,
            group_name=MATCH.group_name,
            group_title=MATCH.group_title,
            internal_layers=MATCH.internal_layers,
            combine_operation=MATCH.combine_operation,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        )),
        salience=100
    )
    def write_component_group_header(
        self,
        layer_index,
        group_name,
        group_title,
        internal_layers,
        combine_operation,
        input_shape,
        output_shape
    ):
        self.write_component_group_header_text(
            layer_index,
            group_name,
            group_title,
            internal_layers,
            combine_operation,
            input_shape,
            output_shape
        )

    @Rule(
        TrainingComponent(
            component_index=MATCH.component_index,
            component_name=MATCH.component_name,
            component_title=MATCH.component_title,
            component_type=MATCH.component_type,
            input_shape=MATCH.input_shape,
            output_shape=MATCH.output_shape,
            status="computed",
            used_during=MATCH.used_during
        ),
        NOT(OutputStatus(
            layer_index=MATCH.component_index,
            output_name="calculation_header",
            status="written"
        )),
        salience=100
    )
    def write_training_component_header(
        self,
        component_index,
        component_name,
        component_title,
        component_type,
        input_shape,
        output_shape,
        used_during
    ):
        self.write_training_component_header_text(
            component_index,
            component_name,
            component_title,
            component_type,
            input_shape,
            output_shape,
            used_during
        )

    @Rule(
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            parameter_name=MATCH.parameter_name,
            value=MATCH.value
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.parameter_name,
            status="parameter_written"
        )),
        salience=90
    )
    def write_layer_parameter(
        self,
        layer_index,
        layer_name,
        parameter_name,
        value
    ):
        self.write_layer_parameter_text(
            layer_index,
            layer_name,
            parameter_name,
            value
        )

    @Rule(
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            question_key=MATCH.question_key,
            choice=MATCH.choice,
            value=MATCH.value,
            status="resolved"
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.question_key,
            status="option_written"
        )),
        salience=88
    )
    def write_option_selection(
        self,
        layer_index,
        layer_name,
        question_key,
        choice,
        value
    ):
        self.write_option_selection_text(
            layer_index,
            layer_name,
            question_key,
            choice,
            value
        )

    @Rule(
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            tensor_name=MATCH.tensor_name,
            role="input",
            dims_count=MATCH.dims_count,
            shape=MATCH.shape,
            data_type=MATCH.data_type,
            source=MATCH.source
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.tensor_name,
            status="tensor_written"
        )),
        salience=86
    )
    def write_input_tensor_info(
        self,
        layer_index,
        layer_name,
        tensor_name,
        dims_count,
        shape,
        data_type,
        source
    ):
        self.write_tensor_info_text(
            "Input tensor / input shape information",
            layer_index,
            layer_name,
            tensor_name,
            "input",
            dims_count,
            shape,
            data_type,
            source
        )

    @Rule(
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            tensor_name=MATCH.tensor_name,
            role=MATCH.role,
            dims_count=MATCH.dims_count,
            shape=MATCH.shape,
            data_type=MATCH.data_type,
            source=MATCH.source
        ),
        TEST(lambda role: role != "input"),
        TEST(lambda role: role != "output"),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.tensor_name,
            status="tensor_written"
        )),
        salience=84
    )
    def write_general_tensor_info(
        self,
        layer_index,
        layer_name,
        tensor_name,
        role,
        dims_count,
        shape,
        data_type,
        source
    ):
        self.write_tensor_info_text(
            "Tensor / metadata information",
            layer_index,
            layer_name,
            tensor_name,
            role,
            dims_count,
            shape,
            data_type,
            source
        )

    @Rule(
        WeightShape(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            weight_name=MATCH.weight_name,
            shape=MATCH.shape
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.weight_name,
            status="weight_written"
        )),
        salience=82
    )
    def write_weight_shape(
        self,
        layer_index,
        layer_name,
        weight_name,
        shape
    ):
        self.write_weight_shape_text(
            layer_index,
            layer_name,
            weight_name,
            shape
        )

    @Rule(
        IntermediateTensor(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            tensor_name=MATCH.tensor_name,
            shape=MATCH.shape,
            data_type=MATCH.data_type,
            source=MATCH.source
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.tensor_name,
            status="intermediate_tensor_written"
        )),
        salience=80
    )
    def write_intermediate_tensor(
        self,
        layer_index,
        layer_name,
        tensor_name,
        shape,
        data_type,
        source
    ):
        self.write_intermediate_tensor_text(
            layer_index,
            layer_name,
            tensor_name,
            shape,
            data_type,
            source
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
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.step_key,
            status="written"
        )),
        salience=70
    )
    def write_calculation_step(
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
        TensorInfo(
            layer_index=MATCH.layer_index,
            layer_name=MATCH.layer_name,
            tensor_name=MATCH.tensor_name,
            role="output",
            dims_count=MATCH.dims_count,
            shape=MATCH.shape,
            data_type=MATCH.data_type,
            source=MATCH.source
        ),
        OutputStatus(
            layer_index=MATCH.layer_index,
            output_name="calculation_header",
            status="written"
        ),
        NOT(OutputStatus(
            layer_index=MATCH.layer_index,
            output_name=MATCH.tensor_name,
            status="tensor_written"
        )),
        salience=40
    )
    def write_output_tensor_info(
        self,
        layer_index,
        layer_name,
        tensor_name,
        dims_count,
        shape,
        data_type,
        source
    ):
        self.write_tensor_info_text(
            "Output tensor / output shape information",
            layer_index,
            layer_name,
            tensor_name,
            "output",
            dims_count,
            shape,
            data_type,
            source
        )
