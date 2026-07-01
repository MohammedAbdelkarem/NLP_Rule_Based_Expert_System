from experta import Fact, Field


class TransformerStage(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    stage_name = Field(str, mandatory=True)
    status = Field(str, mandatory=True)
