from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo
)
from src.prompts import (
    POSITION_WISE_LINEAR_SELECTED,
    ASK_POSITION_WISE_LINEAR_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_INPUT_DIM,
    ASK_POSITION_WISE_LINEAR_OUTPUT_DIM,
    ASK_POSITION_WISE_LINEAR_PURPOSE
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class PositionWiseLinearParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="position_wise_linear"
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind"
        )),
        salience=75
    )
    def ask_first_position_wise_linear_input_kind(self, layer_index):
        print(POSITION_WISE_LINEAR_SELECTED)

        choice = read_text_parameter(ASK_POSITION_WISE_LINEAR_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_position_wise_linear_input_kind_hidden_tensor(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice="1",
            value="encoded_float_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_position_wise_linear_input_kind_raw_token_ids(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_position_wise_linear_input_kind_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid input kind choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_POSITION_WISE_LINEAR_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="position_wise_linear"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="input_data_kind",
            value="encoded_float_tensor",
            status="resolved"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_first_position_wise_linear_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        input_dim = read_int_parameter(ASK_INPUT_DIM)
        output_dim = read_int_parameter(ASK_POSITION_WISE_LINEAR_OUTPUT_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="model_dim",
            value=input_dim
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=output_dim
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="position_wise_linear"
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_batch_size",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.batch_size
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_sequence_length",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.sequence_length
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            tensor_name="output_model_dim",
            role="shape_metadata",
            dims_count=1,
            shape=MATCH.model_dim
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=3,
            data_type="float_hidden_tensor"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim"
        )),
        salience=70
    )
    def ask_position_wise_linear_parameters_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        batch_size,
        sequence_length,
        model_dim
    ):
        output_dim = read_int_parameter(ASK_POSITION_WISE_LINEAR_OUTPUT_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="batch_size",
            value=int(batch_size)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="sequence_length",
            value=int(sequence_length)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="model_dim",
            value=int(model_dim)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=output_dim
        ))

    @Rule(
        AS.output_dim_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=MATCH.output_dim
        ),
        TEST(lambda output_dim: output_dim <= 0),
        salience=69
    )
    def ask_position_wise_linear_output_dim_positive_again(
        self,
        output_dim_fact,
        layer_index,
        output_dim
    ):
        print()
        print("Invalid output dimension. The output dimension must be a positive integer.")

        self.retract(output_dim_fact)

        new_output_dim = read_int_parameter(ASK_POSITION_WISE_LINEAR_OUTPUT_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=new_output_dim
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            parameter_name="output_dim",
            value=MATCH.output_dim
        ),
        TEST(lambda output_dim: output_dim > 0),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose"
        )),
        salience=68
    )
    def ask_position_wise_linear_purpose(self, layer_index, output_dim):
        choice = read_text_parameter(ASK_POSITION_WISE_LINEAR_PURPOSE)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=67
    )
    def resolve_position_wise_linear_hidden_projection(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice="1",
            value="float_hidden_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=67
    )
    def resolve_position_wise_linear_logits_projection(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice="2",
            value="logits",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=66
    )
    def ask_position_wise_linear_purpose_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid Linear purpose choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_POSITION_WISE_LINEAR_PURPOSE)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="position_wise_linear",
            question_key="linear_output_purpose",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
