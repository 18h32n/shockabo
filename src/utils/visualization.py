"""Performance visualization utilities for dashboard display.

This module provides utilities for generating charts, graphs, and other
visualizations for the real-time performance dashboard.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ChartData:
    """Base class for chart data generation."""

    def __init__(self, title: str, chart_type: str):
        """Initialize chart data.

        Args:
            title: Chart title
            chart_type: Type of chart (line, bar, pie, etc.)
        """
        self.title = title
        self.chart_type = chart_type
        self.data: dict[str, Any] = {}
        self.options: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "type": self.chart_type,
            "data": self.data,
            "options": self.options,
        }


class AccuracyTrendChart(ChartData):
    """Chart for displaying accuracy trends over time."""

    def __init__(self, title: str = "Accuracy Trend"):
        """Initialize accuracy trend chart."""
        super().__init__(title, "line")
        self.options = {
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 1.0,
                    "ticks": {"format": {"style": "percent"}},
                },
                "x": {"type": "time", "time": {"unit": "minute"}},
            },
        }

    def add_series(
        self,
        label: str,
        timestamps: list[datetime],
        accuracies: list[float],
        color: str | None = None,
    ):
        """Add a data series to the chart.

        Args:
            label: Series label
            timestamps: List of timestamps
            accuracies: List of accuracy values
            color: Optional color for the series
        """
        if "datasets" not in self.data:
            self.data["datasets"] = []

        dataset = {
            "label": label,
            "data": [
                {"x": ts.isoformat(), "y": acc}
                for ts, acc in zip(timestamps, accuracies, strict=False)
            ],
            "borderWidth": 2,
            "tension": 0.1,
            "fill": False,
        }

        if color:
            dataset["borderColor"] = color
            dataset["backgroundColor"] = color

        self.data["datasets"].append(dataset)


class ProcessingTimeDistribution(ChartData):
    """Chart for displaying processing time distribution."""

    def __init__(self, title: str = "Processing Time Distribution"):
        """Initialize processing time distribution chart."""
        super().__init__(title, "bar")
        self.options = {
            "responsive": True,
            "scales": {
                "y": {"beginAtZero": True},
                "x": {"title": {"display": True, "text": "Time Range (ms)"}},
            },
        }

    def set_distribution(
        self, bins: list[str], counts: list[int], colors: list[str] | None = None
    ):
        """Set the distribution data.

        Args:
            bins: List of bin labels (e.g., "0-50ms", "50-100ms")
            counts: List of counts for each bin
            colors: Optional list of colors for each bin
        """
        self.data = {
            "labels": bins,
            "datasets": [
                {
                    "label": "Task Count",
                    "data": counts,
                    "backgroundColor": colors or ["#3498db"] * len(counts),
                    "borderColor": "#2c3e50",
                    "borderWidth": 1,
                }
            ],
        }


class StrategyComparisonChart(ChartData):
    """Chart for comparing strategy performance."""

    def __init__(self, title: str = "Strategy Performance Comparison"):
        """Initialize strategy comparison chart."""
        super().__init__(title, "radar")
        self.options = {
            "responsive": True,
            "scales": {
                "r": {
                    "beginAtZero": True,
                    "max": 1.0,
                    "ticks": {"stepSize": 0.2},
                }
            },
        }

    def set_comparison(
        self,
        strategies: list[str],
        metrics: dict[str, list[float]],
        metric_labels: list[str],
    ):
        """Set the comparison data.

        Args:
            strategies: List of strategy names
            metrics: Dictionary mapping metric names to values for each strategy
            metric_labels: Labels for each metric axis
        """
        self.data = {"labels": metric_labels, "datasets": []}

        colors = [
            "#e74c3c",
            "#3498db",
            "#2ecc71",
            "#f39c12",
            "#9b59b6",
            "#1abc9c",
            "#34495e",
        ]

        for i, strategy in enumerate(strategies):
            dataset = {
                "label": strategy,
                "data": [metrics[metric][i] for metric in metric_labels],
                "backgroundColor": colors[i % len(colors)] + "33",  # Add transparency
                "borderColor": colors[i % len(colors)],
                "borderWidth": 2,
                "pointBackgroundColor": colors[i % len(colors)],
                "pointBorderColor": "#fff",
                "pointHoverBackgroundColor": "#fff",
                "pointHoverBorderColor": colors[i % len(colors)],
            }
            self.data["datasets"].append(dataset)


class ResourceUtilizationGauge(ChartData):
    """Gauge chart for resource utilization."""

    def __init__(self, resource_type: str):
        """Initialize resource utilization gauge.

        Args:
            resource_type: Type of resource (CPU, Memory, GPU)
        """
        super().__init__(f"{resource_type} Utilization", "gauge")
        self.resource_type = resource_type
        self.options = {
            "responsive": True,
            "min": 0,
            "max": 100,
            "yellowFrom": 60,
            "yellowTo": 80,
            "redFrom": 80,
            "redTo": 100,
            "minorTicks": 5,
        }

    def set_value(self, value: float):
        """Set the gauge value.

        Args:
            value: Utilization percentage (0-100)
        """
        self.data = {"value": round(value, 1), "label": f"{self.resource_type} %"}


class ErrorHeatmap(ChartData):
    """Heatmap for error distribution by category and time."""

    def __init__(self, title: str = "Error Distribution Heatmap"):
        """Initialize error heatmap."""
        super().__init__(title, "heatmap")
        self.options = {
            "responsive": True,
            "colorAxis": {"min": 0, "colors": ["#ffffff", "#ffcccc", "#ff0000"]},
            "datalessRegionColor": "#f5f5f5",
        }

    def set_heatmap_data(
        self,
        error_categories: list[str],
        time_slots: list[str],
        error_counts: list[list[int]],
    ):
        """Set the heatmap data.

        Args:
            error_categories: List of error category names
            time_slots: List of time slot labels
            error_counts: 2D array of error counts [category][time_slot]
        """
        self.data = {
            "rows": error_categories,
            "columns": time_slots,
            "values": error_counts,
        }


class DashboardVisualizer:
    """Service for generating dashboard visualizations."""

    def __init__(self):
        """Initialize the dashboard visualizer."""
        self.logger = structlog.get_logger(__name__).bind(service="visualizer")

    def generate_accuracy_trend(
        self,
        metrics_history: list[dict[str, Any]],
        time_window: timedelta,
        grouping_minutes: int = 5,
    ) -> AccuracyTrendChart:
        """Generate accuracy trend chart.

        Args:
            metrics_history: List of historical metrics
            time_window: Time window to display
            grouping_minutes: Minutes to group data points

        Returns:
            AccuracyTrendChart object
        """
        chart = AccuracyTrendChart("Accuracy Trend")

        # Group metrics by time buckets
        cutoff_time = datetime.now() - time_window
        time_buckets = {}

        for metric in metrics_history:
            if metric["timestamp"] < cutoff_time:
                continue

            # Round to nearest grouping interval
            bucket_time = metric["timestamp"].replace(second=0, microsecond=0)
            bucket_minute = (bucket_time.minute // grouping_minutes) * grouping_minutes
            bucket_time = bucket_time.replace(minute=bucket_minute)

            if bucket_time not in time_buckets:
                time_buckets[bucket_time] = []
            time_buckets[bucket_time].append(metric["accuracy"])

        # Calculate average accuracy per bucket
        timestamps = sorted(time_buckets.keys())
        accuracies = [
            sum(time_buckets[ts]) / len(time_buckets[ts]) for ts in timestamps
        ]

        chart.add_series("Average Accuracy", timestamps, accuracies, "#3498db")

        return chart

    def generate_processing_time_distribution(
        self, processing_times: list[float], num_bins: int = 10
    ) -> ProcessingTimeDistribution:
        """Generate processing time distribution chart.

        Args:
            processing_times: List of processing times in milliseconds
            num_bins: Number of bins for histogram

        Returns:
            ProcessingTimeDistribution object
        """
        chart = ProcessingTimeDistribution()

        if not processing_times:
            chart.set_distribution(["No Data"], [0])
            return chart

        # Create histogram bins
        min_time = min(processing_times)
        max_time = max(processing_times)
        bin_width = (max_time - min_time) / num_bins if max_time > min_time else 1

        bins = []
        counts = []

        for i in range(num_bins):
            bin_start = min_time + i * bin_width
            bin_end = min_time + (i + 1) * bin_width
            bin_label = f"{int(bin_start)}-{int(bin_end)}ms"
            bins.append(bin_label)

            # Count values in this bin
            count = sum(
                1
                for t in processing_times
                if bin_start <= t < bin_end or (i == num_bins - 1 and t == bin_end)
            )
            counts.append(count)

        # Color bins based on performance (green=fast, yellow=medium, red=slow)
        colors = []
        for i in range(num_bins):
            if i < num_bins * 0.3:
                colors.append("#2ecc71")  # Green
            elif i < num_bins * 0.7:
                colors.append("#f39c12")  # Yellow
            else:
                colors.append("#e74c3c")  # Red

        chart.set_distribution(bins, counts, colors)

        return chart

    def generate_strategy_comparison(
        self, strategy_metrics: dict[str, dict[str, float]]
    ) -> StrategyComparisonChart:
        """Generate strategy comparison radar chart.

        Args:
            strategy_metrics: Dictionary mapping strategy names to their metrics

        Returns:
            StrategyComparisonChart object
        """
        chart = StrategyComparisonChart()

        if not strategy_metrics:
            return chart

        strategies = list(strategy_metrics.keys())

        # Normalize metrics to 0-1 scale
        metrics = {
            "accuracy": [],
            "speed": [],
            "efficiency": [],
            "reliability": [],
        }

        for strategy in strategies:
            strategy_data = strategy_metrics[strategy]

            # Accuracy (already 0-1)
            metrics["accuracy"].append(strategy_data.get("average_accuracy", 0.0))

            # Speed (inverse of processing time, normalized)
            avg_time = strategy_data.get("average_processing_time_ms", 1000)
            speed_score = max(0, min(1, 100 / avg_time))  # Fast=100ms -> 1.0
            metrics["speed"].append(speed_score)

            # Efficiency (based on perfect match rate)
            perfect_matches = strategy_data.get("perfect_matches", 0)
            total_tasks = strategy_data.get("tasks_evaluated", 1)
            efficiency = perfect_matches / total_tasks if total_tasks > 0 else 0
            metrics["efficiency"].append(efficiency)

            # Reliability (based on success rate and consistency)
            reliability = strategy_data.get("average_accuracy", 0.0) * 0.7 + efficiency * 0.3
            metrics["reliability"].append(reliability)

        metric_labels = ["Accuracy", "Speed", "Efficiency", "Reliability"]
        chart.set_comparison(strategies, metrics, metric_labels)

        return chart

    def generate_resource_gauges(
        self, resource_utilization: dict[str, float]
    ) -> list[ResourceUtilizationGauge]:
        """Generate resource utilization gauges.

        Args:
            resource_utilization: Dictionary of resource type to utilization percentage

        Returns:
            List of ResourceUtilizationGauge objects
        """
        gauges = []

        for resource_type, utilization in resource_utilization.items():
            gauge = ResourceUtilizationGauge(resource_type.upper())
            gauge.set_value(utilization)
            gauges.append(gauge)

        return gauges

    def generate_error_heatmap(
        self,
        error_history: list[dict[str, Any]],
        time_window: timedelta,
        time_slot_minutes: int = 30,
    ) -> ErrorHeatmap:
        """Generate error distribution heatmap.

        Args:
            error_history: List of error records
            time_window: Time window to analyze
            time_slot_minutes: Minutes per time slot

        Returns:
            ErrorHeatmap object
        """
        chart = ErrorHeatmap()

        if not error_history:
            chart.set_heatmap_data(["No Errors"], ["Current"], [[0]])
            return chart

        # Define error categories
        error_categories = [
            "shape_mismatch",
            "color_error",
            "pattern_error",
            "transformation_error",
            "partial_correct",
            "complete_failure",
        ]

        # Calculate time slots
        now = datetime.now()
        cutoff_time = now - time_window
        num_slots = int(time_window.total_seconds() / (time_slot_minutes * 60))

        time_slots = []
        slot_times = []
        for i in range(num_slots):
            slot_start = cutoff_time + timedelta(minutes=i * time_slot_minutes)
            slot_times.append(slot_start)
            time_slots.append(slot_start.strftime("%H:%M"))

        # Count errors by category and time slot
        error_counts = [[0] * num_slots for _ in error_categories]

        for error in error_history:
            if error["timestamp"] < cutoff_time:
                continue

            # Find time slot
            slot_idx = None
            for i, slot_time in enumerate(slot_times[:-1]):
                if slot_time <= error["timestamp"] < slot_times[i + 1]:
                    slot_idx = i
                    break
            if slot_idx is None and error["timestamp"] >= slot_times[-1]:
                slot_idx = num_slots - 1

            if slot_idx is not None:
                # Find error category
                error_cat = error.get("error_category", "unknown")
                if error_cat in error_categories:
                    cat_idx = error_categories.index(error_cat)
                    error_counts[cat_idx][slot_idx] += 1

        chart.set_heatmap_data(error_categories, time_slots, error_counts)

        return chart

    def generate_dashboard_charts(
        self, dashboard_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate all charts for dashboard display.

        Args:
            dashboard_metrics: Current dashboard metrics

        Returns:
            Dictionary of chart configurations
        """

        # Add various charts based on available data
        # This would be called by the dashboard frontend to get visualization configs

        return {
            "accuracy_trend": self.generate_accuracy_trend(
                dashboard_metrics.get("task_history", []),
                timedelta(hours=1),
            ).to_dict(),
            "processing_distribution": self.generate_processing_time_distribution(
                dashboard_metrics.get("processing_times", [])
            ).to_dict(),
            "strategy_comparison": self.generate_strategy_comparison(
                dashboard_metrics.get("strategy_metrics", {})
            ).to_dict(),
            "resource_gauges": [
                gauge.to_dict()
                for gauge in self.generate_resource_gauges(
                    dashboard_metrics.get("resource_utilization", {})
                )
            ],
            "error_heatmap": self.generate_error_heatmap(
                dashboard_metrics.get("error_history", []),
                timedelta(hours=6),
            ).to_dict(),
        }
