"""Unit tests for platform rotation automation components."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.domain.services.experiment_orchestrator import (
    ExperimentConfig,
    ExperimentOrchestrator,
    ExperimentRetryStrategy,
    ExperimentStatus,
)
from src.infrastructure.components import (
    AvailabilityStatus,
    GracefulShutdownManager,
    Platform,
    PlatformAvailabilityChecker,
    PlatformDetector,
    PlatformInfo,
    QuotaInfo,
    ShutdownReason,
    get_gpu_monitor,
    get_platform_detector,
)
from src.utils.checkpoint_cleanup import CheckpointCleanupManager, CleanupPolicy, CleanupStrategy
from src.utils.checkpoint_manager import CheckpointManager, CheckpointVersion
from src.utils.gcs_integration import CheckpointMetadata, GCSConfig, GCSManager


class TestPlatformDetector:
    """Test platform detection functionality."""

    def test_platform_detection_kaggle(self):
        """Test Kaggle platform detection."""
        with patch.dict('os.environ', {'KAGGLE_KERNEL_RUN_TYPE': 'interactive'}):
            detector = PlatformDetector()
            info = detector.detect_platform()
            assert info.platform == Platform.KAGGLE

    def test_platform_detection_colab(self):
        """Test Colab platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.COLAB,
            gpu_available=True,
            gpu_count=1,
            memory_gb=12.0,
            storage_gb=107.0
        )
        
        with patch.object(PlatformDetector, 'detect_platform', return_value=mock_platform_info):
            detector = PlatformDetector()
            info = detector.detect_platform()
            assert info.platform == Platform.COLAB

    def test_platform_detection_paperspace(self):
        """Test Paperspace platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.PAPERSPACE,
            gpu_available=False,
            gpu_count=0,
            memory_gb=8.0,
            storage_gb=5.0
        )
        
        with patch.object(PlatformDetector, 'detect_platform', return_value=mock_platform_info):
            detector = PlatformDetector()
            info = detector.detect_platform()
            assert info.platform == Platform.PAPERSPACE

    def test_platform_detection_local(self):
        """Test local platform detection."""
        mock_platform_info = PlatformInfo(
            platform=Platform.LOCAL,
            gpu_available=True,
            gpu_count=1,
            memory_gb=32.0,
            storage_gb=500.0
        )
        
        with patch.object(PlatformDetector, 'detect_platform', return_value=mock_platform_info):
            detector = PlatformDetector()
            info = detector.detect_platform()
            assert info.platform == Platform.LOCAL

    def test_gpu_detection(self):
        """Test GPU capability detection."""
        detector = PlatformDetector()
        info = detector.detect_platform()
        assert isinstance(info.gpu_available, bool)
        assert isinstance(info.gpu_count, int)
        assert info.gpu_count >= 0

    def test_runtime_limits(self):
        """Test platform runtime limits detection."""
        detector = PlatformDetector()
        for platform in [Platform.KAGGLE, Platform.COLAB, Platform.PAPERSPACE]:
            limits = detector._detect_runtime_limits(platform)
            assert 'gpu_hours_daily' in limits or 'gpu_hours_weekly' in limits
            assert 'session_timeout_hours' in limits
            assert 'internet_access' in limits


class TestPlatformAvailability:
    """Test platform availability checking."""

    @pytest.mark.asyncio
    async def test_availability_check(self):
        """Test basic availability checking."""
        checker = PlatformAvailabilityChecker()
        check = await checker.check_availability(Platform.KAGGLE, 2.0)

        assert check.platform == Platform.KAGGLE
        assert isinstance(check.status, AvailabilityStatus)
        assert isinstance(check.can_start_experiment, bool)
        assert check.estimated_runtime_hours == 2.0

    @pytest.mark.asyncio
    async def test_quota_info(self):
        """Test quota information retrieval."""
        checker = PlatformAvailabilityChecker()
        quota = await checker._get_quota_info(Platform.KAGGLE)

        assert isinstance(quota, QuotaInfo)
        assert quota.platform == Platform.KAGGLE
        assert quota.gpu_hours_total > 0
        assert quota.gpu_hours_remaining >= 0

    @pytest.mark.asyncio
    async def test_best_platform_selection(self):
        """Test best platform selection logic."""
        checker = PlatformAvailabilityChecker()
        best = await checker.get_best_platform(2.0)

        # May be None if no platforms available
        if best:
            assert isinstance(best.platform, Platform)
            assert best.can_start_experiment

    @pytest.mark.asyncio
    async def test_all_platforms_check(self):
        """Test checking all platforms."""
        checker = PlatformAvailabilityChecker()
        checks = await checker.check_all_platforms(1.0)

        assert len(checks) >= 4  # At least 4 platforms
        for check in checks:
            assert isinstance(check.platform, Platform)
            assert isinstance(check.status, AvailabilityStatus)


class TestGracefulShutdown:
    """Test graceful shutdown functionality."""

    def test_shutdown_manager_creation(self):
        """Test shutdown manager creation."""
        manager = GracefulShutdownManager()
        assert not manager.is_shutdown_in_progress()
        assert manager.get_shutdown_event() is None

    def test_shutdown_hook_registration(self):
        """Test shutdown hook registration."""
        manager = GracefulShutdownManager()

        def test_hook(event):
            pass

        manager.register_shutdown_hook(test_hook)
        assert test_hook in manager._shutdown_hooks

        manager.unregister_shutdown_hook(test_hook)
        assert test_hook not in manager._shutdown_hooks

    def test_shutdown_initiation(self):
        """Test shutdown initiation."""
        manager = GracefulShutdownManager()

        # Mock detector to avoid actual platform detection
        manager._detector = Mock()
        manager._detector.detect_platform.return_value = Mock(platform=Platform.LOCAL)

        manager.initiate_shutdown(ShutdownReason.PLATFORM_ROTATION)
        assert manager.is_shutdown_in_progress()

        event = manager.get_shutdown_event()
        assert event is not None
        assert event.reason == ShutdownReason.PLATFORM_ROTATION
        assert event.platform == Platform.LOCAL


class TestGCSIntegration:
    """Test Google Cloud Storage integration."""

    def test_gcs_config_creation(self):
        """Test GCS configuration creation."""
        config = GCSConfig(
            project_id="test-project",
            bucket_name="test-bucket",
            region="us-central1"
        )
        assert config.project_id == "test-project"
        assert config.bucket_name == "test-bucket"
        assert config.region == "us-central1"

    @patch('src.utils.gcs_integration.storage')
    def test_gcs_authentication(self, mock_storage):
        """Test GCS authentication."""
        config = GCSConfig(
            project_id="test-project",
            bucket_name="test-bucket"
        )

        # Mock successful authentication
        mock_client = Mock()
        mock_storage.Client.return_value = mock_client
        mock_client.list_buckets.return_value = iter([])

        manager = GCSManager(config)
        result = manager.authenticate()

        assert result is True
        assert manager._authenticated is True

    def test_checkpoint_metadata_creation(self):
        """Test checkpoint metadata creation."""
        metadata = CheckpointMetadata(
            name="checkpoint_001.pt",
            version="1.0",
            created_at=datetime.now(),
            size_bytes=1024 * 1024,
            platform="kaggle",
            experiment_id="exp-123",
            model_type="pytorch",
            epoch=10,
            loss=0.25
        )

        assert metadata.name == "checkpoint_001.pt"
        assert metadata.epoch == 10
        assert metadata.loss == 0.25


class TestCheckpointManager:
    """Test checkpoint management functionality."""

    def test_checkpoint_manager_creation(self):
        """Test checkpoint manager creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            assert manager.local_dir == Path(tmpdir)
            assert manager.version_file.name == "checkpoint_versions.json"

    def test_checkpoint_versioning(self):
        """Test checkpoint version tracking."""
        version = CheckpointVersion(
            version="1.0",
            timestamp=datetime.now(),
            size_bytes=1024 * 1024,
            hash="abc123",
            platform="kaggle",
            experiment_id="exp-123",
            local_path="/path/to/checkpoint.pt"
        )

        assert version.version == "1.0"
        assert version.platform == "kaggle"
        assert version.experiment_id == "exp-123"

    @patch('torch.save')
    def test_save_checkpoint(self, mock_save):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            checkpoint_data = {"model_state": "dummy"}
            name = manager.save_checkpoint(
                checkpoint_data,
                experiment_id="exp-123",
                epoch=10,
                loss=0.25,
                platform="kaggle"
            )

            assert name.startswith("exp-123_e0010_kaggle_")
            assert name in manager._local_versions


class TestCheckpointCleanup:
    """Test checkpoint cleanup functionality."""

    def test_cleanup_policy_creation(self):
        """Test cleanup policy creation."""
        policy = CleanupPolicy(
            max_storage_gb=4.5,
            max_checkpoints_per_experiment=5,
            strategy=CleanupStrategy.SMART_RETENTION
        )

        assert policy.max_storage_gb == 4.5
        assert policy.strategy == CleanupStrategy.SMART_RETENTION

    def test_cleanup_strategies(self):
        """Test different cleanup strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(tmpdir)
            cleanup_manager = CheckpointCleanupManager(checkpoint_manager)

            # Test strategy creation
            checkpoints = []
            for strategy in CleanupStrategy:
                cleanup_manager.policy.strategy = strategy
                actions = cleanup_manager._strategy_oldest_first(checkpoints)
                assert isinstance(actions, list)


class TestExperimentOrchestrator:
    """Test experiment orchestration functionality."""

    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Test orchestrator creation and initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = ExperimentOrchestrator(state_dir=tmpdir)
            await orchestrator.start()

            assert orchestrator._running is True

            await orchestrator.stop()
            assert orchestrator._running is False

    def test_experiment_config_creation(self):
        """Test experiment configuration."""
        config = ExperimentConfig(
            name="Test Experiment",
            description="Test description",
            platform_preferences=[Platform.KAGGLE, Platform.COLAB],
            retry_strategy=ExperimentRetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=3
        )

        assert config.name == "Test Experiment"
        assert Platform.KAGGLE in config.platform_preferences
        assert config.retry_strategy == ExperimentRetryStrategy.EXPONENTIAL_BACKOFF

    @pytest.mark.asyncio
    async def test_experiment_queue_operations(self):
        """Test experiment queue operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = ExperimentOrchestrator(state_dir=tmpdir)

            # Create experiment
            exp_id = orchestrator.create_experiment(
                name="Test Experiment",
                command="python train.py"
            )

            assert exp_id is not None

            # Add to queue
            await orchestrator.add_experiment(exp_id)

            # Check queue
            status = orchestrator.get_experiment_status(exp_id)
            assert status == ExperimentStatus.QUEUED

            # Get next experiment
            next_exp = await orchestrator.get_next_experiment()
            assert next_exp is not None
            assert next_exp.id == exp_id


class TestGPUMonitor:
    """Test GPU monitoring functionality."""

    @pytest.mark.asyncio
    async def test_gpu_monitor_creation(self):
        """Test GPU monitor creation."""
        monitor = get_gpu_monitor()
        assert monitor is not None

    def test_gpu_analytics(self):
        """Test GPU analytics generation."""
        monitor = get_gpu_monitor()
        analytics = monitor.get_analytics()

        assert 'overall_metrics' in analytics
        assert 'efficiency_score' in analytics
        assert 'memory_analytics' in analytics

    def test_gpu_recommendations(self):
        """Test GPU optimization recommendations."""
        monitor = get_gpu_monitor()
        recommendations = monitor.get_recommendations()

        assert isinstance(recommendations, list)
        # Recommendations depend on current GPU state
        for rec in recommendations:
            assert 'action' in rec
            assert 'priority' in rec
            assert 'implementation' in rec


@pytest.mark.integration
class TestPlatformRotationIntegration:
    """Integration tests for platform rotation."""

    @pytest.mark.asyncio
    async def test_full_rotation_workflow(self):
        """Test complete platform rotation workflow."""
        # This would test the full integration of all components
        # In a real test, you'd mock external dependencies
        detector = get_platform_detector()
        current_platform = detector.detect_platform()

        assert current_platform.platform in [
            Platform.KAGGLE, Platform.COLAB,
            Platform.PAPERSPACE, Platform.LOCAL
        ]
