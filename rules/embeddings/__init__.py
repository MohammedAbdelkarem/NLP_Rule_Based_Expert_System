from rules.embeddings.token_embedding import (
    TokenEmbeddingParameterRules,
    TokenEmbeddingCalculationRules,
    TokenEmbeddingOutputRules,
)
from rules.embeddings.pretrained_static_embedding import (
    PretrainedStaticEmbeddingParameterRules,
    PretrainedStaticEmbeddingOptionResolutionRules,
    PretrainedStaticEmbeddingCalculationRules,
    PretrainedStaticEmbeddingOutputRules,
)
from rules.embeddings.compatibility import (
    TokenEmbeddingCompatibilityRules,
    PretrainedStaticEmbeddingCompatibilityRules,
)
from rules.embeddings.feature_extraction.count_based_embedding import (
    CountBasedEmbeddingParameterRules,
    CountBasedEmbeddingCalculationRules,
    CountBasedEmbeddingOutputRules,
)
from rules.embeddings.feature_extraction.co_occurrence_matrix import (
    CoOccurrenceMatrixParameterRules,
    CoOccurrenceMatrixCalculationRules,
    CoOccurrenceMatrixOutputRules,
)
from rules.embeddings.similarity.cosine_similarity import (
    CosineSimilarityParameterRules,
    CosineSimilarityCalculationRules,
    CosineSimilarityOutputRules,
)
from rules.embeddings.training.skipgram_sgns import (
    SkipGramSGNSParameterRules,
    SkipGramSGNSCalculationRules,
    SkipGramSGNSOutputRules,
)
from rules.embeddings.analysis.visualization import (
    EmbeddingVisualizationParameterRules,
    EmbeddingVisualizationCalculationRules,
    EmbeddingVisualizationOutputRules,
)
from rules.embeddings.analysis.bias_analysis import (
    EmbeddingBiasAnalysisParameterRules,
    EmbeddingBiasAnalysisCalculationRules,
    EmbeddingBiasAnalysisOutputRules,
)
from rules.embeddings.analysis.evaluation import (
    EmbeddingEvaluationParameterRules,
    EmbeddingEvaluationCalculationRules,
    EmbeddingEvaluationOutputRules,
)

from rules.embeddings.pooling.embedding_to_classifier_pooling import (
    EmbeddingToClassifierPoolingParameterRules,
    EmbeddingToClassifierPoolingCalculationRules,
    EmbeddingToClassifierPoolingOutputRules,
)
from rules.embeddings.pooling.compatibility import (
    EmbeddingPoolingCompatibilityRules,
)
