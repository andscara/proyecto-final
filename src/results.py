from dataclasses import dataclass
from forecasting.forecast_manager import ForecastFitResult
from metrics import ForecastMetricType
from profiling.segmentation_result import SegmentationResult


@dataclass
class ProfileForecastingMatching:
    segmentation_result: SegmentationResult
    model_results: ForecastFitResult

    def overall_score_by_metric(self, metric: ForecastMetricType) -> float:
        total = 0.0
        # we need to sum the min scores for each cluster and being the min of all models
        for _, model_dict in self.model_results.items():
            min_score = min(
                scores[metric] 
                for _, scores in model_dict.items()
                if metric in scores
            )
            total += min_score
        return total
    
@dataclass
class TrainingResult:
    best_matchings: dict[ForecastMetricType, ProfileForecastingMatching]

    def get_best_matching_for_metric(self, metric: ForecastMetricType) -> ProfileForecastingMatching:
        matching = self.best_matchings.get(metric, None)
        if matching is None:
            raise ValueError(f"No best matching found for metric {metric}.")
        return matching
    

@dataclass
class GlobalPrediction:
    metric: ForecastMetricType
    