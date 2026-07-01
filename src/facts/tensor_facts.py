from experta import Fact, Field


class TensorInfo(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    tensor_name = Field(str, mandatory=True)
    role = Field(str, mandatory=True)
    dims_count = Field(int, mandatory=True)
    shape = Field(str, mandatory=True)
    data_type = Field(str, mandatory=True)
    source = Field(str, mandatory=True)


class IntermediateTensor(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    tensor_name = Field(str, mandatory=True)
    shape = Field(str, mandatory=True)
    data_type = Field(str, mandatory=True)
    source = Field(str, mandatory=True)


class WeightShape(Fact):
    layer_index = Field(int, mandatory=True)
    layer_name = Field(str, mandatory=True)
    weight_name = Field(str, mandatory=True)
    shape = Field(str, mandatory=True)
