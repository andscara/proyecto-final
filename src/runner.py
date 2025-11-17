import logging
from forecasting.forecast_manager import ForecastManager
from input import Input
from metrics import ForecastMetricType
from profiling.profiling_manager import ProfilingManager
from results import ProfileForecastingMatching, TrainingResult
from storage.storage import BaseStorage


def train(
    input: Input,
    logger: logging.Logger,
    storage: BaseStorage,
    profiling_manager: ProfilingManager,
    forecast_manager: ForecastManager
):
    logger.info("Starting training phase.")
    storage.initialize()
    # Need to get all the profiler results and pass them to forecasting models
    data = input.data_source.read_data() # TODO: Change this to read in batches
    logger.info("Starting profiling phase.")
    segmentation_results = profiling_manager.run_profilers(data)
    profile_forecast_matchings: list[ProfileForecastingMatching] = []
    # For each segmentation result, we need to run the forecast models
    for segmentation_result in segmentation_results:
        storage.save_segmentation_result(segmentation_result)
        logger.info(f"Processing profiling result for profiler {segmentation_result.id}")
        model_fit_results = forecast_manager.fitAll(
            input = input, 
            average_data = segmentation_result.average_data,
            cluster_names = list(set(segmentation_result.mapping.values()))
        )
        logger.info(f"Best models for profiler {segmentation_result.id}: {model_fit_results}")
        profile_forecast_matching = ProfileForecastingMatching(
            segmentation_result=segmentation_result,
            model_results=model_fit_results
        )
        profile_forecast_matchings.append(profile_forecast_matching)
    # Now we need to calculate the best ProfileForecastingMatching per metric type
    best_matchings: dict[ForecastMetricType, ProfileForecastingMatching] = {}
    for metric in ForecastMetricType:
        best_matching = min(
            profile_forecast_matchings,
            key=lambda matching: matching.overall_score_by_metric(metric)
        )
        logger.info(f"Best profiling-forecasting match for metric {metric}: Profiler {best_matching.segmentation_result.id} with score {best_matching.overall_score_by_metric(metric)}")
        best_matchings[metric] = best_matching
    # Store all the data related to the profiling and forecasting models
    training_result = TrainingResult(best_matchings=best_matchings)
    logger.info("Saving training result to storage.")
    storage.save_training_result(training_result)
    logger.info("Training phase completed.")
    

def predict(
    input: Input,
    metric: ForecastMetricType,
    storage: BaseStorage,
    profiling_manager: ProfilingManager,
    forecast_manager: ForecastManager
):
    ...  # Implementation of the predict function goes here