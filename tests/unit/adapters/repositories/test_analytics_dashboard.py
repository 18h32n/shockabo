"""Unit tests for analytics dashboard."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.adapters.repositories.analytics_dashboard import AnalyticsDashboard, TimeSeriesMetric
from src.adapters.repositories.program_cache import (
    CacheStatistics,
    PatternAnalysis,
    ProgramCache,
)
from src.adapters.repositories.program_cache_config import ProgramCacheConfig


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_cache():
    """Create mock program cache."""
    cache = Mock(spec=ProgramCache)

    # Mock statistics
    cache.get_statistics.return_value = CacheStatistics(
        total_programs=100,
        successful_programs=85,
        unique_programs=90,
        total_size_bytes=50 * 1024 * 1024,  # 50MB
        cache_hit_rate=0.75,
        average_access_frequency=2.5,
        most_successful_patterns=[
            "Operation sequence: rotate -> flip",
            "Operation sequence: fill -> mask",
            "Structure pattern: rotate×2, flip×1"
        ],
        task_type_distribution={'training': 70, 'evaluation': 20, 'test': 10},
        generation_distribution={0: 20, 1: 30, 2: 25, 3: 15, 4: 10}
    )

    # Mock patterns
    cache.get_patterns.return_value = {
        'pat1': PatternAnalysis(
            pattern_id='pat1',
            pattern_type='sequence',
            pattern_description='Operation sequence: rotate -> flip',
            operation_sequence=['rotate', 'flip'],
            frequency=50,
            success_rate=0.92,
            program_ids=['prog1', 'prog2', 'prog3'],
            created_at=datetime.now(),
            last_updated=datetime.now()
        ),
        'pat2': PatternAnalysis(
            pattern_id='pat2',
            pattern_type='structure',
            pattern_description='Structure pattern: fill×2, mask×1',
            operation_sequence=[],
            frequency=30,
            success_rate=0.85,
            program_ids=['prog4', 'prog5'],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    }

    return cache


@pytest.fixture
def analytics_dashboard(mock_cache, temp_cache_dir):
    """Create analytics dashboard instance."""
    dashboard = AnalyticsDashboard(
        cache=mock_cache,
        output_dir=Path(temp_cache_dir) / "analytics"
    )
    return dashboard


class TestAnalyticsDashboard:
    """Test AnalyticsDashboard functionality."""

    def test_initialization(self, analytics_dashboard):
        """Test dashboard initialization."""
        assert analytics_dashboard.cache is not None
        assert analytics_dashboard.output_dir.exists()
        assert analytics_dashboard.metrics_file.name == "metrics_history.json"
        assert isinstance(analytics_dashboard.metrics_history, list)

    def test_collect_metrics(self, analytics_dashboard):
        """Test metrics collection."""
        metrics = analytics_dashboard.collect_metrics()

        assert 'timestamp' in metrics
        assert 'cache_stats' in metrics
        assert metrics['cache_stats']['total_programs'] == 100
        assert metrics['cache_stats']['cache_hit_rate'] == 0.75
        assert metrics['cache_stats']['total_size_mb'] == 50.0

        assert 'task_distribution' in metrics
        assert metrics['task_distribution']['training'] == 70

        assert 'pattern_count' in metrics
        assert metrics['pattern_count'] == 2

        assert 'top_patterns' in metrics
        assert len(metrics['top_patterns']) == 3

    def test_metrics_history_persistence(self, analytics_dashboard):
        """Test metrics history is saved and loaded."""
        # Collect metrics
        metrics1 = analytics_dashboard.collect_metrics()

        # Verify it's saved
        assert analytics_dashboard.metrics_file.exists()
        assert len(analytics_dashboard.metrics_history) == 1

        # Create new dashboard instance
        dashboard2 = AnalyticsDashboard(
            cache=analytics_dashboard.cache,
            output_dir=analytics_dashboard.output_dir
        )

        # Should load previous metrics
        assert len(dashboard2.metrics_history) == 1
        assert dashboard2.metrics_history[0]['timestamp'] == metrics1['timestamp']

    def test_generate_html_report(self, analytics_dashboard):
        """Test HTML report generation."""
        # Add some metrics history
        analytics_dashboard.collect_metrics()

        report_path = analytics_dashboard.generate_analytics_report(
            format='html',
            include_visuals=False  # Skip visuals for unit test
        )

        assert report_path is not None
        assert Path(report_path).exists()

        # Check content
        with open(report_path) as f:
            content = f.read()

        assert '<title>Program Cache Analytics Dashboard</title>' in content
        assert 'Total Programs' in content
        assert '100' in content  # Total programs value
        assert 'Cache Hit Rate' in content
        assert '75.0%' in content  # Hit rate value

    def test_generate_json_report(self, analytics_dashboard):
        """Test JSON report generation."""
        analytics_dashboard.collect_metrics()

        report_path = analytics_dashboard.generate_analytics_report(
            format='json',
            include_visuals=False
        )

        assert report_path is not None
        assert Path(report_path).exists()

        # Check content
        with open(report_path) as f:
            data = json.load(f)

        assert 'timestamp' in data
        assert 'cache_stats' in data
        assert data['cache_stats']['total_programs'] == 100

    def test_export_metrics_csv(self, analytics_dashboard):
        """Test CSV export functionality."""
        # Add multiple metrics
        for i in range(3):
            analytics_dashboard.collect_metrics()

        csv_path = analytics_dashboard.export_metrics_csv()

        assert csv_path
        assert Path(csv_path).exists()

        # Check CSV content
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert 'timestamp' in rows[0]
        assert 'total_programs' in rows[0]
        assert 'cache_hit_rate' in rows[0]
        assert rows[0]['total_programs'] == '100'

    @patch('src.adapters.repositories.analytics_dashboard.plt')
    def test_plot_cache_metrics_timeline(self, mock_plt, analytics_dashboard):
        """Test cache metrics timeline plot (matplotlib)."""
        # Add metrics over time
        base_time = datetime.now()
        for i in range(5):
            with patch('src.adapters.repositories.analytics_dashboard.datetime') as mock_dt:
                mock_dt.now.return_value = base_time + timedelta(hours=i)
                mock_dt.fromisoformat = datetime.fromisoformat
                analytics_dashboard.collect_metrics()

        # Test plot creation
        fig = analytics_dashboard._plot_cache_metrics_timeline()

        assert fig is not None
        mock_plt.subplots.assert_called_once()

    @patch('src.adapters.repositories.analytics_dashboard.plt')
    def test_plot_pattern_success_rates(self, mock_plt, analytics_dashboard):
        """Test pattern success rates plot."""
        # Ensure patterns exist
        analytics_dashboard.cache.get_patterns()

        fig = analytics_dashboard._plot_pattern_success_rates()

        assert fig is not None
        mock_plt.subplots.assert_called_once()

    def test_plot_task_distribution(self, analytics_dashboard):
        """Test task distribution plot."""
        # Collect metrics first
        analytics_dashboard.collect_metrics()

        with patch('src.adapters.repositories.analytics_dashboard.plt'):
            fig = analytics_dashboard._plot_task_distribution()
            assert fig is not None

    def test_plot_generation_distribution(self, analytics_dashboard):
        """Test generation distribution plot."""
        # Collect metrics first
        analytics_dashboard.collect_metrics()

        with patch('src.adapters.repositories.analytics_dashboard.plt'):
            fig = analytics_dashboard._plot_generation_distribution()
            assert fig is not None

    def test_invalid_report_format(self, analytics_dashboard):
        """Test error handling for invalid report format."""
        with pytest.raises(ValueError) as exc_info:
            analytics_dashboard.generate_analytics_report(format='invalid')

        assert "Unsupported format" in str(exc_info.value)

    def test_time_series_metric_dataclass(self):
        """Test TimeSeriesMetric dataclass."""
        metric = TimeSeriesMetric(
            timestamp=datetime.now(),
            value=0.85,
            label="Hit Rate"
        )

        assert isinstance(metric.timestamp, datetime)
        assert metric.value == 0.85
        assert metric.label == "Hit Rate"


class TestProgramCacheAnalyticsIntegration:
    """Test analytics integration with ProgramCache."""

    def test_analytics_enabled_config(self, temp_cache_dir):
        """Test analytics is enabled with config."""
        config = ProgramCacheConfig()
        config.analytics.enable_analytics = True
        config.storage.cache_dir = temp_cache_dir

        cache = ProgramCache(config=config)

        assert cache.analytics_dashboard is not None
        assert isinstance(cache.analytics_dashboard, AnalyticsDashboard)

    def test_analytics_disabled_config(self, temp_cache_dir):
        """Test analytics is disabled with config."""
        config = ProgramCacheConfig()
        config.analytics.enable_analytics = False
        config.storage.cache_dir = temp_cache_dir

        cache = ProgramCache(config=config)

        assert cache.analytics_dashboard is None

    def test_generate_report_via_cache(self, temp_cache_dir):
        """Test generating report via cache interface."""
        config = ProgramCacheConfig()
        config.analytics.enable_analytics = True
        config.storage.cache_dir = temp_cache_dir

        cache = ProgramCache(config=config)

        # Should not crash even with empty cache
        report = cache.generate_analytics_report(format='json')

        if report:
            assert Path(report).exists()

    def test_collect_metrics_via_cache(self, temp_cache_dir):
        """Test collecting metrics via cache interface."""
        config = ProgramCacheConfig()
        config.analytics.enable_analytics = True
        config.storage.cache_dir = temp_cache_dir

        cache = ProgramCache(config=config)

        metrics = cache.collect_analytics_metrics()

        assert metrics is not None
        assert 'timestamp' in metrics
        assert 'cache_stats' in metrics
