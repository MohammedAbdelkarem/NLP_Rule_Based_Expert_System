from experta import AS, MATCH, NOT, Rule, TEST

from src.facts import LayerParameter, OptionSelection, SelectedLayer, TensorInfo
from src.prompts import (
    ASK_BATCH_SIZE,
    ASK_CLASSIFIER_INPUT_KIND,
    ASK_CLASSIFIER_MODE,
    ASK_FEATURE_DIM,
    ASK_HIDDEN_DIM,
    ASK_NUM_CLASSES,
    FEEDFORWARD_CLASSIFIER_HEAD_SELECTED,
)


from src.utils.console_control import read_int_parameter, read_text_parameter
class FeedforwardClassifierHeadParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="feedforward_classifier_head",
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
        )),
        salience=75,
    )
    def ask_first_classifier_input_kind(self, layer_index):
        print(FEEDFORWARD_CLASSIFIER_HEAD_SELECTED)
        choice = read_text_parameter(ASK_CLASSIFIER_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_classifier_feature_vector_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="1",
            value="feature_vector_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_classifier_sequence_tensor_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="2",
            value="sequence_hidden_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="3",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_classifier_raw_token_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice="3",
            value="raw_token_ids",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=73,
    )
    def ask_classifier_input_kind_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid input kind choice. Please choose 1, 2, or 3.")
        self.retract(option_fact)
        new_choice = read_text_parameter(ASK_CLASSIFIER_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="feedforward_classifier_head",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_input_kind",
            value="feature_vector_tensor",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="batch_size",
        )),
        salience=70,
    )
    def ask_first_classifier_vector_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        input_dim = read_int_parameter(ASK_FEATURE_DIM)
        hidden_dim = read_int_parameter(ASK_HIDDEN_DIM)
        classifier_choice = read_text_parameter(ASK_CLASSIFIER_MODE)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="batch_size",
            value=batch_size,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="model_dim",
            value=input_dim,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="hidden_dim",
            value=hidden_dim,
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice=classifier_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=MATCH.previous_layer_index,
            layer_name="feedforward_classifier_head",
        ),
        TensorInfo(
            layer_index=MATCH.previous_layer_index,
            role="output",
            dims_count=2,
            data_type="float_hidden_tensor",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="hidden_dim",
        )),
        salience=70,
    )
    def ask_classifier_after_previous_parameters(self, layer_index):
        hidden_dim = read_int_parameter(ASK_HIDDEN_DIM)
        classifier_choice = read_text_parameter(ASK_CLASSIFIER_MODE)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="hidden_dim",
            value=hidden_dim,
        ))
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice=classifier_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=69,
    )
    def resolve_binary_classifier_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice="1",
            value="binary_classification",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=69,
    )
    def resolve_multiclass_classifier_mode(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice="2",
            value="multiclass_classification",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=68,
    )
    def ask_classifier_mode_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid classifier mode. Please choose 1 or 2.")
        self.retract(option_fact)
        new_choice = read_text_parameter(ASK_CLASSIFIER_MODE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            value="binary_classification",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="num_classes",
        )),
        salience=67,
    )
    def set_binary_classifier_output_count(self, layer_index):
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="num_classes",
            value=1,
        ))

    @Rule(
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            question_key="classifier_mode",
            value="multiclass_classification",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="num_classes",
        )),
        salience=67,
    )
    def ask_multiclass_output_count(self, layer_index):
        num_classes = read_int_parameter(ASK_NUM_CLASSES)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="feedforward_classifier_head",
            parameter_name="num_classes",
            value=num_classes,
        ))
