from experta import Fact, Field


class MenuChoice(Fact):
    choice = Field(str, mandatory=True)


class SelectedLayer(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    layer_title = Field(str, mandatory=True)
    source_name = Field(str, mandatory=True)


class UnsupportedLayer(Fact):
    choice = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)


class LayerParameter(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    parameter_name = Field(str, mandatory=True)
    value = Field(int, mandatory=True)


class TensorShape(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    shape_role = Field(str, mandatory=True)
    dims_count = Field(int, mandatory=True)
    shape = Field(str, mandatory=True)


class WeightShape(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    weight_name = Field(str, mandatory=True)
    shape = Field(str, mandatory=True)


class Layer(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    layer_title = Field(str, mandatory=True)
    input_shape = Field(str, mandatory=True)
    output_shape = Field(str, mandatory=True)


class LayerExplanation(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    text = Field(str, mandatory=True)


class CompatibilityResult(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    status = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)


class TransformationSuggestion(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    from_shape = Field(str, mandatory=True)
    to_shape = Field(str, mandatory=True)
    method = Field(str, mandatory=True)


class OutputStatus(Fact):
    layer_index = Field(int, mandatory=True)
    output_name = Field(str, mandatory=True)
    status = Field(str, mandatory=True)


class ContinueChoice(Fact):
    choice = Field(str, mandatory=True)


class SessionStatus(Fact):
    status = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)


class CalculationStep(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    step_key = Field(str, mandatory=True)
    previous_step_key = Field(str, mandatory=True)
    step_order = Field(int, mandatory=True)
    title = Field(str, mandatory=True)
    formula = Field(str, mandatory=True)
    substitution = Field(str, mandatory=True)
    result = Field(str, mandatory=True)
