from experta import Fact, Field


class OutputStatus(Fact):
    layer_index = Field(int, mandatory=True)
    output_name = Field(str, mandatory=True)
    status = Field(str, mandatory=True)


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
