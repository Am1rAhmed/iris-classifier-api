from prometheus_client import Counter

prediction_counter = Counter(
    "iris_predictions_total",
    "Total number of successful predictions, labeled by predicted species and API version",
    ["species", "api_version"]
)