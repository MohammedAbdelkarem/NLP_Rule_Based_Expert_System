from experta import AS, MATCH, NOT, Rule, TEST

from facts import LayerParameter, OptionSelection, SelectedLayer
from prompts import (
    ASK_BATCH_SIZE,
    ASK_EMBEDDING_DIM,
    ASK_POOLING_INPUT_KIND,
    ASK_SEQUENCE_LENGTH,
    EMBEDDING_POOLING_SELECTED,
)


from utils.console_control import read_int_parameter, read_text_parameter
class EmbeddingToClassifierPoolingParameterRules:
    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="embedding_to_classifier_pooling",
        ),
        NOT(OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
        )),
        salience=75,
    )
    def ask_first_pooling_input_kind(self, layer_index):
        print(EMBEDDING_POOLING_SELECTED)
        choice = read_text_parameter(ASK_POOLING_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pooling_sequence_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="1",
            value="sequence_embedding_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pooling_feature_vector_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="2",
            value="feature_vector_tensor",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="3",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pooling_raw_token_input(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice="3",
            value="raw_token_ids",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        salience=73,
    )
    def ask_pooling_input_kind_again(self, option_fact, layer_index, choice):
        print()
        print("Invalid pooling input kind. Please choose 1, 2, or 3.")
        self.retract(option_fact)
        new_choice = read_text_parameter(ASK_POOLING_INPUT_KIND)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            choice=new_choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        SelectedLayer(
            layer_index=MATCH.layer_index,
            previous_layer_index=0,
            layer_name="embedding_to_classifier_pooling",
        ),
        OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            question_key="pooling_input_kind",
            value="sequence_embedding_tensor",
            status="resolved",
        ),
        NOT(LayerParameter(
            layer_index=MATCH.layer_index,
            layer_name="embedding_to_classifier_pooling",
            parameter_name="batch_size",
        )),
        salience=70,
    )
    def ask_first_pooling_shape_parameters(self, layer_index):
        batch_size = read_int_parameter(ASK_BATCH_SIZE)
        sequence_length = read_int_parameter(ASK_SEQUENCE_LENGTH)
        embedding_dim = read_int_parameter(ASK_EMBEDDING_DIM)
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            parameter_name="batch_size",
            value=batch_size,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            parameter_name="sequence_length",
            value=sequence_length,
        ))
        self.declare(LayerParameter(
            layer_index=layer_index,
            layer_name="embedding_to_classifier_pooling",
            parameter_name="model_dim",
            value=embedding_dim,
        ))
