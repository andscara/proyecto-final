from dataclasses import dataclass
from forecasting.forecast_model import ForecastModelID
from metrics import ForecastMetricType
from profiling.profiling_result import SegmentationResult


@dataclass
class ProfileForecastingMatching:
    profile: SegmentationResult
    model_results: dict[ForecastModelID, dict[ForecastMetricType, float]]

    def overall_score_by_metric(self, metric: ForecastMetricType) -> float:
        total = 0.0
        for model_result in self.model_results.values():
            score = model_result.get(metric, None)
            if score is None:
                raise ValueError(f"Metric {metric} not found in model results.")
            total += score
        return total / len(self.model_results)
    
@dataclass
class TrainingResult:
    best_matchings: dict[ForecastMetricType, ProfileForecastingMatching]

    def get_best_matching_for_metric(self, metric: ForecastMetricType) -> ProfileForecastingMatching:
        matching = self.best_matchings.get(metric, None)
        if matching is None:
            raise ValueError(f"No best matching found for metric {metric}.")
        return matching