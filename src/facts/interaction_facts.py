from experta import Fact, Field


class MenuChoice(Fact):
    layer_index = Field(int, mandatory=True)
    previous_layer_index = Field(int, mandatory=True)
    choice = Field(str, mandatory=True)


class SelectedLayer(Fact):
    layer_index = Field(int, mandatory=True)
    previous_layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    layer_title = Field(str, mandatory=True)
    source_name = Field(str, mandatory=True)


class LayerParameter(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    parameter_name = Field(str, mandatory=True)
    value = Field(int, mandatory=True)


class OptionSelection(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    question_key = Field(str, mandatory=True)
    choice = Field(str, mandatory=True)
    value = Field(str, mandatory=True)
    status = Field(str, mandatory=True)


class RollbackRequest(Fact):
    layer_index = Field(int, mandatory=True)
    previous_layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    reason = Field(str, mandatory=True)
    message = Field(str, mandatory=True)


class ContinueChoice(Fact):
    after_layer_index = Field(int, mandatory=True)
    choice = Field(str, mandatory=True)
