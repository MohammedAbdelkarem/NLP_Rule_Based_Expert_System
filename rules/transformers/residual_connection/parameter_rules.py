from experta import Rule, MATCH, NOT, AS, TEST

from facts import (
    SelectedLayer,
    LayerParameter,
    TensorInfo,
    TransformationSuggestion
)
from prompts import (
    RESIDUAL_CONNECTION_SELECTED,
    ASK_RESIDUAL_SOURCE_LAYER_INDEX
)


from utils.console_control import read_int_parameter, read_text_parameter
class ResidualConnectionParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            data_type="float_hidden_tensor"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index"
        )),
        salience=70
    )
    def ask_residual_source_layer_index(self, layer_index, previous_layer_index):
        print(RESIDUAL_CONNECTION_SELECTED)

        source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=source_layer_index
        ))

    @Rule(
        AS.source_layer_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        TEST(lambda source_layer_index: source_layer_index <= 0),
        salience=69
    )
    def ask_residual_source_positive_again(
        self,
        source_layer_fact,
        layer_index,
        source_layer_index
    ):
        print()
        print("Invalid residual source layer index. Choose an earlier positive layer index.")

        self.retract(source_layer_fact)

        new_source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=new_source_layer_index
        ))

    @Rule(
        AS.source_layer_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
        ),
        TEST(lambda source_layer_index, previous_layer_index: source_layer_index >= previous_layer_index),
        salience=68
    )
    def ask_residual_source_earlier_again(
        self,
        source_layer_fact,
        layer_index,
        previous_layer_index,
        source_layer_index
    ):
        print()
        print("Invalid residual source. The source tensor must come from a layer before the previous layer.")
        print(f"Previous layer index: {previous_layer_index}")

        self.retract(source_layer_fact)

        new_source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=new_source_layer_index
        ))

    @Rule(
        AS.source_layer_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        TEST(lambda source_layer_index: source_layer_index > 0),
        NOT(TensorInfo(
            layer_index=MATCH.source_layer_index,
            role="output"
        )),
        salience=67
    )
    def ask_residual_source_existing_output_again(
        self,
        source_layer_fact,
        layer_index,
        source_layer_index
    ):
        print()
        print("Invalid residual source. No output tensor was found for this layer index.")
        print(f"Selected source layer index: {source_layer_index}")

        self.retract(source_layer_fact)

        new_source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=new_source_layer_index
        ))

    @Rule(
        AS.source_layer_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type=MATCH.previous_data_type
        ),
        TensorInfo(
            layer_index=MATCH.source_layer_index,
            role="output",
            shape=MATCH.source_shape,
            data_type=MATCH.source_data_type
        ),
        TEST(lambda previous_data_type, source_data_type: previous_data_type != source_data_type),
        salience=66
    )
    def ask_residual_source_data_type_again(
        self,
        source_layer_fact,
        layer_index,
        source_layer_index,
        previous_layer_index,
        previous_shape,
        source_shape,
        previous_data_type,
        source_data_type
    ):
        print()
        print("Invalid residual source. Residual addition requires compatible data types.")
        print(f"Previous output data type: {previous_data_type}")
        print(f"Source output data type: {source_data_type}")

        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape=f"previous={previous_shape}, source={source_shape}",
            to_shape="two tensors with the same shape and compatible data type",
            method="Choose another residual source tensor or add an adapter to make both tensors compatible."
        ))

        self.retract(source_layer_fact)

        new_source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=new_source_layer_index
        ))

    @Rule(
        AS.source_layer_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=MATCH.source_layer_index
        ),
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="residual_connection"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            shape=MATCH.previous_shape,
            data_type=MATCH.data_type
        ),
        TensorInfo(
            layer_index=MATCH.source_layer_index,
            role="output",
            shape=MATCH.source_shape,
            data_type=MATCH.data_type
        ),
        TEST(lambda previous_shape, source_shape: previous_shape != source_shape),
        salience=65
    )
    def ask_residual_source_shape_again(
        self,
        source_layer_fact,
        layer_index,
        source_layer_index,
        previous_layer_index,
        previous_shape,
        source_shape,
        data_type
    ):
        print()
        print("Invalid residual source. Residual addition requires equal tensor shapes.")
        print(f"Previous output shape: {previous_shape}")
        print(f"Source output shape: {source_shape}")

        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="residual_connection",
            from_shape=f"previous={previous_shape}, source={source_shape}",
            to_shape="two tensors with the same shape",
            method="Choose another residual source tensor or add an adapter/projection to match the shapes."
        ))

        self.retract(source_layer_fact)

        new_source_layer_index = read_int_parameter(ASK_RESIDUAL_SOURCE_LAYER_INDEX)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="residual_connection",
            parameter_name="residual_source_layer_index",
            value=new_source_layer_index
        ))
