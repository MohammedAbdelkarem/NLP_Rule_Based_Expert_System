from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    TransformationSuggestion
)
from src.prompts import (
    FEED_FORWARD_SELECTED,
    ASK_FEED_FORWARD_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_MODEL_DIM,
    ASK_FEED_FORWARD_DIM,
    ASK_FEED_FORWARD_ACTIVATION
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class FeedForwardParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="feed_forward"
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind"
        )),
        salience=75
    )
    def ask_first_feed_forward_input_kind(self, layer_index):
        print(FEED_FORWARD_SELECTED)

        choice = read_text_parameter(ASK_FEED_FORWARD_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_feed_forward_input_kind_hidden_tensor(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice="1",
            value="encoded_float_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_feed_forward_input_kind_raw_token_ids(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_feed_forward_input_kind_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid input kind choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_FEED_FORWARD_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="feed_forward"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="input_data_kind",
            value="encoded_float_tensor",
            status="resolved"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_first_feed_forward_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        model_dim = read_int_parameter(ASK_MODEL_DIM)
        feed_forward_dim = read_int_parameter(ASK_FEED_FORWARD_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="model_dim",
            value=model_dim
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=feed_forward_dim
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feed_forward"
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
            layer_name="feed_forward",
            parameter_name="feed_forward_dim"
        )),
        salience=70
    )
    def ask_feed_forward_parameters_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        batch_size,
        sequence_length,
        model_dim
    ):
        feed_forward_dim = read_int_parameter(ASK_FEED_FORWARD_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="batch_size",
            value=int(batch_size)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="sequence_length",
            value=int(sequence_length)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="model_dim",
            value=int(model_dim)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=feed_forward_dim
        ))

    @Rule(
        AS.feed_forward_dim_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=MATCH.feed_forward_dim
        ),
        TEST(lambda feed_forward_dim: feed_forward_dim <= 0),
        salience=69
    )
    def ask_feed_forward_dim_positive_again(
        self,
        feed_forward_dim_fact,
        layer_index,
        feed_forward_dim
    ):
        print()
        print("Invalid feed-forward dimension. d_ff must be a positive integer.")

        self.retract(feed_forward_dim_fact)

        new_feed_forward_dim = read_int_parameter(ASK_FEED_FORWARD_DIM)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=new_feed_forward_dim
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward"
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            parameter_name="feed_forward_dim",
            value=MATCH.feed_forward_dim
        ),
        TEST(lambda feed_forward_dim: feed_forward_dim > 0),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation"
        )),
        salience=68
    )
    def ask_feed_forward_activation(self, layer_index, feed_forward_dim):
        choice = read_text_parameter(ASK_FEED_FORWARD_ACTIVATION)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=67
    )
    def resolve_feed_forward_activation_relu(self, option_fact, layer_index):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice="1",
            value="relu",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=67
    )
    def resolve_feed_forward_activation_gelu(self, option_fact, layer_index):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice="2",
            value="gelu",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=66
    )
    def ask_feed_forward_activation_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid activation choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_FEED_FORWARD_ACTIVATION)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feed_forward",
            question_key="ffn_activation",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))
