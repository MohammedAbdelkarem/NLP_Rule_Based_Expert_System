from experta import Fact, Field


class Layer(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    layer_title = Field(str, mandatory=True)
    input_shape = Field(str, mandatory=True)
    output_shape = Field(str, mandatory=True)


class ComponentGroup(Fact):
    group_index = Field(int, mandatory=True)
    group_name = Field(str, mandatory=True)
    group_title = Field(str, mandatory=True)
    internal_layers = Field(str, mandatory=True)
    combine_operation = Field(str, mandatory=True)
    input_shape = Field(str, mandatory=True)
    output_shape = Field(str, mandatory=True)
    status = Field(str, mandatory=True)


class LayerMetadata(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    layer_kind = Field(str, mandatory=True)
    layer_role = Field(str, mandatory=True)
    repeat_policy = Field(str, mandatory=True)
    consecutive_repeat_policy = Field(str, mandatory=True)
    terminal_status = Field(str, mandatory=True)


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
