"""
Tests for the execution profiler functionality.

Tests CPU profiling, memory tracking, and profiling data export capabilities.
"""

import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.execution_profiler import (
    ProfilingConfig,
    ProfilingManager,
    create_profiling_manager,
    get_resource_usage_snapshot,
)
from src.adapters.strategies.profiling_exporter import ProfilingDataExporter
from src.adapters.strategies.python_transpiler import (
    ExecutionMetrics,
    MemoryAllocationData,
    ProfileData,
)
from src.infrastructure.config import TranspilerSandboxConfig


class TestProfilingConfig:
    """Test profiling configuration."""

    def test_default_config(self):
        """Test default profiling configuration values."""
        config = ProfilingConfig()

        assert not config.cpu_profiling_enabled
        assert not config.memory_tracking_enabled
        assert config.resource_monitoring_enabled
        assert not config.profile_builtin_functions
        assert config.max_profile_functions == 50
        assert config.tracemalloc_enabled
        assert config.memory_snapshot_interval == 0.1
        assert config.max_traced_allocations == 1000
        assert config.monitor_peak_memory
        assert config.monitor_cpu_times
        assert not config.export_raw_profile
        assert not config.export_memory_traces

    def test_custom_config(self):
        """Test custom profiling configuration."""
        config = ProfilingConfig(
            cpu_profiling_enabled=True,
            memory_tracking_enabled=True,
            max_profile_functions=100,
            export_raw_profile=True
        )

        assert config.cpu_profiling_enabled
        assert config.memory_tracking_enabled
        assert config.max_profile_functions == 100
        assert config.export_raw_profile


class TestProfilingManager:
    """Test profiling manager functionality."""

    def test_profiling_manager_initialization(self):
        """Test profiling manager initialization."""
        config = ProfilingConfig(cpu_profiling_enabled=True)
        manager = ProfilingManager(config)

        assert manager.config.cpu_profiling_enabled
        assert manager._profiler is None
        assert manager._memory_start_snapshot is None
        assert manager._memory_peak_mb == 0.0
        assert manager._memory_traces == []

    def test_profile_execution_context_manager(self):
        """Test profiling context manager."""
        config = ProfilingConfig(resource_monitoring_enabled=True)
        manager = ProfilingManager(config)

        with manager.profile_execution():
            # Simulate some work
            time.sleep(0.01)
            result = sum(range(1000))

        # Should have started timing
        assert manager._start_time > 0

    @patch('cProfile.Profile')
    def test_cpu_profiling_enabled(self, mock_profile_class):
        """Test CPU profiling functionality."""
        mock_profiler = Mock()
        mock_profile_class.return_value = mock_profiler

        config = ProfilingConfig(cpu_profiling_enabled=True)
        manager = ProfilingManager(config)

        with manager.profile_execution():
            pass

        # Should have created and used profiler
        mock_profile_class.assert_called_once()
        mock_profiler.enable.assert_called_once()
        mock_profiler.disable.assert_called_once()

    @patch('tracemalloc.is_tracing')
    @patch('tracemalloc.start')
    @patch('tracemalloc.take_snapshot')
    def test_memory_tracking_enabled(self, mock_snapshot, mock_start, mock_is_tracing):
        """Test memory tracking functionality."""
        mock_is_tracing.return_value = False
        mock_snapshot.return_value = Mock()

        config = ProfilingConfig(memory_tracking_enabled=True)
        manager = ProfilingManager(config)

        with manager.profile_execution():
            pass

        # Should have started tracemalloc and taken snapshot
        mock_start.assert_called_once()
        mock_snapshot.assert_called()

    def test_get_profile_data_disabled(self):
        """Test getting profile data when profiling is disabled."""
        config = ProfilingConfig(cpu_profiling_enabled=False)
        manager = ProfilingManager(config)

        profile_data = manager.get_profile_data()

        assert not profile_data.enabled
        assert profile_data.total_calls == 0
        assert profile_data.total_time == 0.0
        assert len(profile_data.top_functions) == 0

    @patch('cProfile.Profile')
    @patch('pstats.Stats')
    def test_get_profile_data_enabled(self, mock_stats_class, mock_profile_class):
        """Test getting profile data when CPU profiling is enabled."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profile_class.return_value = mock_profiler

        mock_stats = Mock()
        mock_stats.total_calls = 100
        mock_stats.prim_calls = 80
        mock_stats.total_tt = 0.5
        mock_stats.stats = {
            ('test.py', 10, 'test_func'): (5, 5, 0.1, 0.2, {}),
            ('main.py', 20, 'main_func'): (10, 8, 0.3, 0.4, {})
        }
        mock_stats_class.return_value = mock_stats

        config = ProfilingConfig(cpu_profiling_enabled=True, max_profile_functions=10)
        manager = ProfilingManager(config)

        with manager.profile_execution():
            pass

        profile_data = manager.get_profile_data()

        assert profile_data.enabled
        assert profile_data.total_calls == 100
        assert profile_data.primitive_calls == 80
        assert profile_data.total_time == 0.5
        assert len(profile_data.top_functions) == 2

    def test_get_memory_allocation_data_disabled(self):
        """Test getting memory allocation data when tracking is disabled."""
        config = ProfilingConfig(memory_tracking_enabled=False)
        manager = ProfilingManager(config)

        memory_data = manager.get_memory_allocation_data()

        assert not memory_data.enabled
        assert memory_data.peak_memory_mb == 0.0
        assert memory_data.allocation_count == 0
        assert len(memory_data.top_allocators) == 0

    @patch('src.adapters.strategies.execution_profiler.HAS_PSUTIL', True)
    @patch('psutil.Process')
    def test_get_current_memory_mb_with_psutil(self, mock_process_class):
        """Test memory measurement with psutil."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 50 * 1024 * 1024  # 50 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        config = ProfilingConfig()
        manager = ProfilingManager(config)

        memory_mb = manager._get_current_memory_mb()

        assert memory_mb == 50.0
        assert manager._memory_peak_mb == 50.0

    def test_export_profile_data_cpu_only(self):
        """Test exporting CPU profile data."""
        config = ProfilingConfig(cpu_profiling_enabled=True, export_raw_profile=True)
        manager = ProfilingManager(config)

        # Mock the profiler with proper stats structure
        with patch('cProfile.Profile') as mock_profile_class:
            mock_profiler = Mock()
            # Mock the stats attribute to be an empty dict to avoid iteration errors
            mock_profiler.stats = {}
            mock_profile_class.return_value = mock_profiler
            manager._profiler = mock_profiler

            # Mock pstats.Stats to avoid issues with the mock profiler
            with patch('pstats.Stats') as mock_stats:
                mock_stats_instance = Mock()
                mock_stats.return_value = mock_stats_instance
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    exported_files = manager.export_profile_data(
                        f"{temp_dir}/test_profile"
                    )

                    # Should export CPU profile
                    assert 'cpu_profile' in exported_files
                    mock_profiler.dump_stats.assert_called_once()


class TestProfilingDataExporter:
    """Test profiling data export functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ProfilingDataExporter(self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)

    def test_exporter_initialization(self):
        """Test exporter initialization."""
        assert self.exporter.output_dir.exists()
        assert self.exporter.output_dir.is_dir()

    def test_export_execution_metrics_basic(self):
        """Test exporting basic execution metrics."""
        metrics = ExecutionMetrics(
            execution_time_ms=100.5,
            memory_used_mb=25.3,
            operation_timings={'op1': 30.0, 'op2': 60.0},
            slow_operations=[('op2', 60.0)]
        )

        exported_files = self.exporter.export_execution_metrics(
            metrics, 'test_program', '20241201_120000'
        )

        # Should export JSON metrics file
        assert 'metrics_json' in exported_files
        json_file = Path(exported_files['metrics_json'])
        assert json_file.exists()

        # Verify JSON content
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        assert data['execution_time_ms'] == 100.5
        assert data['memory_used_mb'] == 25.3
        assert len(data['operation_timings']) == 2
        assert len(data['slow_operations']) == 1

        # Should export CSV timings file
        assert 'timings_csv' in exported_files
        csv_file = Path(exported_files['timings_csv'])
        assert csv_file.exists()

        # Verify CSV content
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['Operation'] == 'op1'
        assert float(rows[0]['Time_ms']) == 30.0
        assert rows[0]['Is_Slow'] == 'False'
        assert rows[1]['Operation'] == 'op2'
        assert rows[1]['Is_Slow'] == 'True'

    def test_export_execution_metrics_with_profiling(self):
        """Test exporting metrics with profiling data."""
        profile_data = ProfileData(
            enabled=True,
            total_calls=1000,
            primitive_calls=800,
            total_time=0.5,
            cumulative_time=0.5,
            top_functions=[
                ('test_func', 100, 0.1, 0.2),
                ('main_func', 50, 0.2, 0.3)
            ]
        )

        memory_data = MemoryAllocationData(
            enabled=True,
            peak_memory_mb=50.0,
            current_memory_mb=45.0,
            allocation_count=200,
            deallocation_count=150,
            net_allocations=50,
            top_allocators=[
                ('numpy.array', 20.0),
                ('list.append', 5.0)
            ]
        )

        metrics = ExecutionMetrics(
            execution_time_ms=100.5,
            memory_used_mb=25.3,
            operation_timings={'op1': 30.0},
            slow_operations=[],
            profile_data=profile_data,
            memory_allocation_data=memory_data
        )

        exported_files = self.exporter.export_execution_metrics(
            metrics, 'test_program'
        )

        # Should export additional profiling files
        assert 'metrics_json' in exported_files
        assert 'timings_csv' in exported_files

        # Verify JSON includes profiling data
        json_file = Path(exported_files['metrics_json'])
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        assert 'profile_data' in data
        assert data['profile_data']['enabled']
        assert data['profile_data']['total_calls'] == 1000
        assert len(data['profile_data']['top_functions']) == 2

        assert 'memory_allocation_data' in data
        assert data['memory_allocation_data']['enabled']
        assert data['memory_allocation_data']['peak_memory_mb'] == 50.0
        assert len(data['memory_allocation_data']['top_allocators']) == 2

    def test_export_batch_metrics(self):
        """Test exporting batch metrics."""
        metrics_list = [
            ExecutionMetrics(
                execution_time_ms=100.0,
                memory_used_mb=20.0,
                operation_timings={'op1': 50.0},
                slow_operations=[('op1', 50.0)]
            ),
            ExecutionMetrics(
                execution_time_ms=200.0,
                memory_used_mb=30.0,
                operation_timings={'op2': 30.0, 'op3': 40.0},
                slow_operations=[]
            )
        ]

        program_ids = ['prog1', 'prog2']
        exported_files = self.exporter.export_batch_metrics(
            metrics_list, program_ids, 'test_batch'
        )

        # Should export summary and detailed files
        assert 'batch_summary' in exported_files
        assert 'batch_detailed' in exported_files

        # Verify summary JSON
        summary_file = Path(exported_files['batch_summary'])
        assert summary_file.exists()

        with open(summary_file, encoding='utf-8') as f:
            summary = json.load(f)

        assert summary['batch_size'] == 2
        assert summary['execution_time_stats']['avg_ms'] == 150.0
        assert summary['memory_usage_stats']['avg_mb'] == 25.0

        # Verify detailed CSV
        detailed_file = Path(exported_files['batch_detailed'])
        assert detailed_file.exists()

        with open(detailed_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['Program_ID'] == 'prog1'
        assert float(rows[0]['Execution_Time_ms']) == 100.0
        assert int(rows[0]['Slow_Operation_Count']) == 1
        assert rows[1]['Program_ID'] == 'prog2'
        assert float(rows[1]['Execution_Time_ms']) == 200.0
        assert int(rows[1]['Slow_Operation_Count']) == 0

    def test_create_performance_report(self):
        """Test creating performance report."""
        metrics_list = [
            ExecutionMetrics(
                execution_time_ms=100.0,
                memory_used_mb=20.0,
                operation_timings={'op1': 30.0},
                slow_operations=[('op1', 60.0)]
            )
        ]

        program_ids = ['test_program']
        report_file = self.exporter.create_performance_report(
            metrics_list, program_ids, 'test_report'
        )

        report_path = Path(report_file)
        assert report_path.exists()

        # Verify report content
        with open(report_path, encoding='utf-8') as f:
            content = f.read()

        assert '# Performance Analysis Report' in content
        assert 'Total Programs Analyzed: 1' in content
        assert 'Average: 100.00 ms' in content
        assert 'test_program: op1 (60.00 ms)' in content


class TestCreateProfilingManager:
    """Test profiling manager factory function."""

    def test_create_profiling_manager_default(self):
        """Test creating profiling manager with defaults."""
        manager = create_profiling_manager()

        assert not manager.config.cpu_profiling_enabled
        assert not manager.config.memory_tracking_enabled
        assert manager.config.resource_monitoring_enabled

    def test_create_profiling_manager_with_options(self):
        """Test creating profiling manager with custom options."""
        transpiler_config = TranspilerSandboxConfig()
        manager = create_profiling_manager(
            cpu_profiling=True,
            memory_tracking=True,
            resource_monitoring=False,
            transpiler_config=transpiler_config
        )

        assert manager.config.cpu_profiling_enabled
        assert manager.config.memory_tracking_enabled
        assert not manager.config.resource_monitoring_enabled


class TestGetResourceUsageSnapshot:
    """Test resource usage snapshot functionality."""

    @patch('src.adapters.strategies.execution_profiler.HAS_PSUTIL', True)
    @patch('psutil.Process')
    def test_get_resource_usage_snapshot_with_psutil(self, mock_process_class):
        """Test resource usage snapshot with psutil."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 25.5
        mock_process_class.return_value = mock_process

        snapshot = get_resource_usage_snapshot()

        assert snapshot['available']
        assert snapshot['memory_mb'] == 100.0
        assert snapshot['cpu_percent'] == 25.5
        assert 'timestamp' in snapshot

    @pytest.mark.skipif(sys.platform == 'win32', reason="resource module not available on Windows")
    @patch('src.adapters.strategies.execution_profiler.HAS_PSUTIL', False)
    @patch('sys.platform', 'linux')
    @patch('resource.getrusage')
    def test_get_resource_usage_snapshot_with_resource(self, mock_getrusage):
        """Test resource usage snapshot with resource module."""
        mock_usage = Mock()
        mock_usage.ru_maxrss = 50 * 1024  # 50 MB in KB
        mock_getrusage.return_value = mock_usage

        snapshot = get_resource_usage_snapshot()

        assert snapshot['available']
        assert snapshot['memory_mb'] == 50.0
        assert 'timestamp' in snapshot

    @patch('src.adapters.strategies.execution_profiler.HAS_PSUTIL', False)
    @patch('sys.platform', 'win32')
    def test_get_resource_usage_snapshot_fallback(self):
        """Test resource usage snapshot fallback."""
        snapshot = get_resource_usage_snapshot()

        assert not snapshot['available']
        assert snapshot['memory_mb'] == 0.0
        assert 'timestamp' in snapshot


if __name__ == '__main__':
    pytest.main([__file__])
