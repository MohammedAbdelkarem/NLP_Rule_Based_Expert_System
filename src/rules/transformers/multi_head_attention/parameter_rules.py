from experta import Rule, MATCH, NOT, AS, TEST

from src.facts import (
    SelectedLayer,
    LayerParameter,
    OptionSelection,
    TensorInfo,
    TransformationSuggestion
)
from src.prompts import (
    MULTI_HEAD_ATTENTION_SELECTED,
    ASK_MULTI_HEAD_ATTENTION_INPUT_KIND,
    ASK_BATCH_SIZE,
    ASK_SEQUENCE_LENGTH,
    ASK_MODEL_DIM,
    ASK_NUM_HEADS
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class MultiHeadAttentionParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multi_head_attention"
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind"
        )),
        salience=75
    )
    def ask_first_multi_head_attention_input_kind(self, layer_index):
        print(MULTI_HEAD_ATTENTION_SELECTED)

        choice = read_text_parameter(ASK_MULTI_HEAD_ATTENTION_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice=choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice="1",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_multi_head_attention_input_kind_hidden_tensor(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice="1",
            value="encoded_float_tensor",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice="2",
            value="unknown",
            status="raw"
        ),
        salience=74
    )
    def resolve_multi_head_attention_input_kind_raw_token_ids(
        self,
        option_fact,
        layer_index
    ):
        self.retract(option_fact)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice="2",
            value="raw_token_ids",
            status="resolved"
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw"
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=73
    )
    def ask_multi_head_attention_input_kind_again(
        self,
        option_fact,
        layer_index,
        choice
    ):
        print()
        print("Invalid input kind choice. Please choose 1 or 2.")

        self.retract(option_fact)

        new_choice = read_text_parameter(ASK_MULTI_HEAD_ATTENTION_INPUT_KIND)

        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            choice=new_choice,
            value="unknown",
            status="raw"
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="multi_head_attention"
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            question_key="input_data_kind",
            value="encoded_float_tensor",
            status="resolved"
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="batch_size"
        )),
        salience=70
    )
    def ask_first_multi_head_attention_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        model_dim = read_int_parameter(ASK_MODEL_DIM)
        num_heads = read_int_parameter(ASK_NUM_HEADS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="batch_size",
            value=batch_size
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="sequence_length",
            value=sequence_length
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="model_dim",
            value=model_dim
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=num_heads
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="multi_head_attention"
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
            layer_name="multi_head_attention",
            parameter_name="num_heads"
        )),
        salience=70
    )
    def ask_multi_head_attention_parameters_after_previous_layer(
        self,
        layer_index,
        previous_layer_index,
        batch_size,
        sequence_length,
        model_dim
    ):
        num_heads = read_int_parameter(ASK_NUM_HEADS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="batch_size",
            value=int(batch_size)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="sequence_length",
            value=int(sequence_length)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="model_dim",
            value=int(model_dim)
        ))

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=num_heads
        ))

    @Rule(
        AS.num_heads_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=MATCH.num_heads
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        TEST(lambda num_heads: num_heads <= 0),
        salience=69
    )
    def ask_multi_head_attention_num_heads_positive_again(
        self,
        num_heads_fact,
        layer_index,
        num_heads,
        model_dim
    ):
        print()
        print("Invalid number of heads. num_heads must be a positive integer.")

        self.retract(num_heads_fact)

        new_num_heads = read_int_parameter(ASK_NUM_HEADS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=new_num_heads
        ))

    @Rule(
        AS.num_heads_fact << LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=MATCH.num_heads
        ),
        LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="multi_head_attention",
            parameter_name="model_dim",
            value=MATCH.model_dim
        ),
        TEST(lambda model_dim, num_heads: num_heads > 0),
        TEST(lambda model_dim, num_heads: model_dim % num_heads != 0),
        salience=68
    )
    def ask_multi_head_attention_num_heads_divisibility_again(
        self,
        num_heads_fact,
        layer_index,
        num_heads,
        model_dim
    ):
        print()
        print("Invalid number of heads. model_dim must be divisible by num_heads.")
        print(f"Current values: model_dim = {model_dim}, num_heads = {num_heads}")

        self.declare(TransformationSuggestion(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            from_shape=f"model_dim={model_dim}, num_heads={num_heads}",
            to_shape="model_dim divisible by num_heads",
            method="Choose another num_heads value or add a projection layer later to change model_dim."
        ))

        self.retract(num_heads_fact)

        new_num_heads = read_int_parameter(ASK_NUM_HEADS)

        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="multi_head_attention",
            parameter_name="num_heads",
            value=new_num_heads
        ))
