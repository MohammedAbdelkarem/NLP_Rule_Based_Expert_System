import collections
import collections.abc

# Compatibility patch for Experta/frozendict on newer Python versions.
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

from experta import KnowledgeEngine

from src.rules.common.start_rules import StartRules
from src.rules.common.menu_rules import MenuRules
from src.rules.common.continue_rules import ContinueRules
from src.rules.common.rollback_rules import RollbackRules
from src.rules.common.calculation_trace_rules import CalculationTraceRules
from src.rules.common.summary_rules import SummaryRules

from src.rules.transformers.compatibility import (
    InputOrderCompatibilityRules,
    ConsecutiveLayerCompatibilityRules,
    DataTypeCompatibilityRules,
    TensorAcceptanceCompatibilityRules
)

from src.rules.transformers.input_encoding import (
    InputEncodingParameterRules,
    InputEncodingTokenEmbeddingRules,
    InputEncodingPositionalEmbeddingRules,
    InputEncodingSummationRules,
    InputEncodingOutputRules
)

from src.rules.transformers.layer_norm import (
    LayerNormParameterRules,
    LayerNormFirstCalculationRules,
    LayerNormPreviousCalculationRules,
    LayerNormOutputRules
)

from src.rules.transformers.multi_head_attention import (
    MultiHeadAttentionParameterRules,
    MultiHeadAttentionQKVProjectionRules,
    MultiHeadAttentionScoreRules,
    MultiHeadAttentionSoftmaxRules,
    MultiHeadAttentionWeightedSumRules,
    MultiHeadAttentionOutputProjectionRules,
    MultiHeadAttentionOutputRules
)

from src.rules.transformers.residual_connection import (
    ResidualConnectionParameterRules,
    ResidualConnectionCalculationRules,
    ResidualConnectionOutputRules
)


from src.rules.transformers.feed_forward import (
    FeedForwardParameterRules,
    FeedForwardFirstProjectionRules,
    FeedForwardActivationRules,
    FeedForwardSecondProjectionRules,
    FeedForwardOutputRules
)

from src.rules.activations import (
    ReLUParameterRules,
    ReLUCalculationRules,
    ReLUOutputRules,
    GELUParameterRules,
    GELUCalculationRules,
    GELUOutputRules,
    SoftmaxParameterRules,
    SoftmaxCalculationRules,
    SoftmaxOutputRules
)

from src.rules.activations.compatibility import (
    ActivationCompatibilityRules,
    SoftmaxActivationCompatibilityRules
)

from src.rules.projections.compatibility import LinearProjectionCompatibilityRules

from src.rules.projections import (
    PositionWiseLinearParameterRules,
    PositionWiseLinearCalculationRules,
    PositionWiseLinearOutputRules
)

from src.rules.speech import (
    AudioInputCompatibilityRules,
    FramingWindowingCompatibilityRules,
    STFTSpectrumCompatibilityRules,
    MelFilterbankCompatibilityRules,
    LogMelSpectrogramCompatibilityRules,
    MFCCCompatibilityRules,
    DeltaFeatureCompatibilityRules,
    FeatureNormalizationCompatibilityRules,
    AudioInputParameterRules,
    AudioInputCalculationRules,
    AudioInputOutputRules,
    FramingWindowingParameterRules,
    FramingWindowingCalculationRules,
    FramingWindowingOutputRules,
    STFTSpectrumParameterRules,
    STFTSpectrumCalculationRules,
    STFTSpectrumOutputRules,
    MelFilterbankParameterRules,
    MelFilterbankCalculationRules,
    MelFilterbankOutputRules,
    LogMelSpectrogramParameterRules,
    LogMelSpectrogramCalculationRules,
    LogMelSpectrogramOutputRules,
    MFCCParameterRules,
    MFCCCalculationRules,
    MFCCOutputRules,
    DeltaFeatureParameterRules,
    DeltaFeatureCalculationRules,
    DeltaFeatureAdapterRules,
    DeltaFeatureOutputRules,
    FeatureNormalizationParameterRules,
    FeatureNormalizationCalculationRules,
    FeatureNormalizationOutputRules
)


class NeuralNetworkExpertSystem(
    StartRules,
    MenuRules,
    ContinueRules,
    RollbackRules,
    CalculationTraceRules,
    SummaryRules,
    InputOrderCompatibilityRules,
    ConsecutiveLayerCompatibilityRules,
    DataTypeCompatibilityRules,
    TensorAcceptanceCompatibilityRules,
    ActivationCompatibilityRules,
    LinearProjectionCompatibilityRules,
    SoftmaxActivationCompatibilityRules,
    AudioInputCompatibilityRules,
    FramingWindowingCompatibilityRules,
    STFTSpectrumCompatibilityRules,
    MelFilterbankCompatibilityRules,
    LogMelSpectrogramCompatibilityRules,
    MFCCCompatibilityRules,
    DeltaFeatureCompatibilityRules,
    FeatureNormalizationCompatibilityRules,
    AudioInputParameterRules,
    AudioInputCalculationRules,
    AudioInputOutputRules,
    FramingWindowingParameterRules,
    FramingWindowingCalculationRules,
    FramingWindowingOutputRules,
    STFTSpectrumParameterRules,
    STFTSpectrumCalculationRules,
    STFTSpectrumOutputRules,
    MelFilterbankParameterRules,
    MelFilterbankCalculationRules,
    MelFilterbankOutputRules,
    LogMelSpectrogramParameterRules,
    LogMelSpectrogramCalculationRules,
    LogMelSpectrogramOutputRules,
    MFCCParameterRules,
    MFCCCalculationRules,
    MFCCOutputRules,
    DeltaFeatureParameterRules,
    DeltaFeatureCalculationRules,
    DeltaFeatureAdapterRules,
    DeltaFeatureOutputRules,
    FeatureNormalizationParameterRules,
    FeatureNormalizationCalculationRules,
    FeatureNormalizationOutputRules,
    InputEncodingParameterRules,
    InputEncodingTokenEmbeddingRules,
    InputEncodingPositionalEmbeddingRules,
    InputEncodingSummationRules,
    InputEncodingOutputRules,
    LayerNormParameterRules,
    LayerNormFirstCalculationRules,
    LayerNormPreviousCalculationRules,
    LayerNormOutputRules,
    MultiHeadAttentionParameterRules,
    MultiHeadAttentionQKVProjectionRules,
    MultiHeadAttentionScoreRules,
    MultiHeadAttentionSoftmaxRules,
    MultiHeadAttentionWeightedSumRules,
    MultiHeadAttentionOutputProjectionRules,
    MultiHeadAttentionOutputRules,
    ResidualConnectionParameterRules,
    ResidualConnectionCalculationRules,
    ResidualConnectionOutputRules,
    FeedForwardParameterRules,
    FeedForwardFirstProjectionRules,
    FeedForwardActivationRules,
    FeedForwardSecondProjectionRules,
    FeedForwardOutputRules,
    ReLUParameterRules,
    ReLUCalculationRules,
    ReLUOutputRules,
    GELUParameterRules,
    GELUCalculationRules,
    GELUOutputRules,
    PositionWiseLinearParameterRules,
    PositionWiseLinearCalculationRules,
    PositionWiseLinearOutputRules,
    SoftmaxParameterRules,
    SoftmaxCalculationRules,
    SoftmaxOutputRules,
    KnowledgeEngine
):
    engine_name = "neural_network_expert_system"
