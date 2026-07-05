from experta import AS, MATCH, Rule, TEST

from src.facts import OptionSelection
from src.prompts import (
    ASK_PRETRAINED_EMBEDDING_SOURCE,
    ASK_PRETRAINED_EMBEDDING_TRAINING_MODE,
)


class PretrainedStaticEmbeddingOptionResolutionRules:
    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pretrained_source_word2vec(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="1",
            value="word2vec",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pretrained_source_glove(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="2",
            value="glove",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="3",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pretrained_source_fasttext(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="3",
            value="fasttext",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="4",
            value="unknown",
            status="raw",
        ),
        salience=74,
    )
    def resolve_pretrained_source_custom(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice="4",
            value="custom_pretrained_matrix",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        TEST(lambda choice: choice != "3"),
        TEST(lambda choice: choice != "4"),
        salience=73,
    )
    def ask_pretrained_source_again(self, option_fact, layer_index):
        print("Invalid pretrained embedding source. Please choose 1, 2, 3, or 4.")
        self.retract(option_fact)
        choice = input(ASK_PRETRAINED_EMBEDDING_SOURCE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_source",
            choice=choice,
            value="unknown",
            status="raw",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice="1",
            value="unknown",
            status="raw",
        ),
        salience=71,
    )
    def resolve_pretrained_training_mode_frozen(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice="1",
            value="frozen",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice="2",
            value="unknown",
            status="raw",
        ),
        salience=71,
    )
    def resolve_pretrained_training_mode_fine_tunable(self, option_fact, layer_index):
        self.retract(option_fact)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice="2",
            value="fine_tunable",
            status="resolved",
        ))

    @Rule(
        AS.option_fact << OptionSelection(
            layer_index=MATCH.layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice=MATCH.choice,
            value="unknown",
            status="raw",
        ),
        TEST(lambda choice: choice != "1"),
        TEST(lambda choice: choice != "2"),
        salience=70,
    )
    def ask_pretrained_training_mode_again(self, option_fact, layer_index):
        print("Invalid training mode. Please choose 1 or 2.")
        self.retract(option_fact)
        choice = input(ASK_PRETRAINED_EMBEDDING_TRAINING_MODE)
        self.declare(OptionSelection(
            layer_index=layer_index,
            layer_name="pretrained_static_embedding",
            question_key="pretrained_embedding_training_mode",
            choice=choice,
            value="unknown",
            status="raw",
        ))
