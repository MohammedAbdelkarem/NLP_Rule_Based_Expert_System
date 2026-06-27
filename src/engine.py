from experta import KnowledgeEngine

from src.rules.common.start_rules import StartRules
from src.rules.common.menu_rules import MenuRules
from src.rules.common.summary_rules import SummaryRules
from src.rules.common.calculation_trace_rules import CalculationTraceRules

from src.rules.source_transformers.token_embedding_rules import TokenEmbeddingRules


class NeuralNetworkExpertSystem(
    StartRules,
    MenuRules,
    SummaryRules,
    CalculationTraceRules,
    TokenEmbeddingRules,
    KnowledgeEngine
):
    engine_name = "neural_network_expert_system"
