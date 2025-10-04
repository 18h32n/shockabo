"""Unit tests for false negative detection in pruning system.

Tests the detection, tracking, and alerting of false negatives (good programs
that were incorrectly pruned).
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.adapters.strategies.false_negative_detector import (
    FalseNegativeAlert,
    FalseNegativeDetector,
)
from src.domain.evaluation_models import EvaluationResult
from src.domain.models import (
    PruningDecision,
    PruningResult,
)


class TestFalseNegativeDetector:
    """Test false negative detection functionality."""

    @pytest.fixture
    def detector(self):
        """Create test false negative detector."""
        return FalseNegativeDetector(
            alert_threshold=0.05,  # 5% threshold
            window_size_minutes=60,
            min_samples_for_alert=100,
        )

    def create_pruning_result(
        self,
        program_id: str,
        decision: PruningDecision,
        confidence: float = 0.5
    ) -> PruningResult:
        """Create test pruning result."""
        return PruningResult(
            program_id=program_id,
            decision=decision,
            confidence_score=confidence,
            pruning_time_ms=5.0,
            rejection_reason="Test rejection" if decision != PruningDecision.ACCEPT else None,
        )

    def create_evaluation_result(
        self,
        program_id: str,
        success: bool,
        accuracy: float = 0.0
    ) -> EvaluationResult:
        """Create test evaluation result."""
        result = Mock(spec=EvaluationResult)
        result.task_id = program_id
        result.final_accuracy = accuracy
        result.best_attempt = Mock(pixel_accuracy=Mock(accuracy=accuracy)) if success else None
        return result

    def test_true_negative_detection(self, detector):
        """Test correct identification of true negatives."""
        # Program was pruned
        pruning_result = self.create_pruning_result(
            "prog1",
            PruningDecision.REJECT_PATTERN,
        )

        # Program actually failed evaluation
        eval_result = self.create_evaluation_result(
            "prog1",
            success=False,
            accuracy=0.0
        )

        # Record results
        is_false_negative = detector.record_validation_result(
            pruning_result,
            eval_result
        )

        assert not is_false_negative  # Correctly rejected
        assert detector.get_metrics()["true_negatives"] == 1
        assert detector.get_metrics()["false_negatives"] == 0

    def test_false_negative_detection(self, detector):
        """Test detection of false negatives."""
        # Program was pruned
        pruning_result = self.create_pruning_result(
            "prog2",
            PruningDecision.REJECT_CONFIDENCE,
            confidence=0.4
        )

        # Program actually succeeded with high accuracy
        eval_result = self.create_evaluation_result(
            "prog2",
            success=True,
            accuracy=0.9
        )

        # Record results
        is_false_negative = detector.record_validation_result(
            pruning_result,
            eval_result
        )

        assert is_false_negative  # Incorrectly rejected
        assert detector.get_metrics()["false_negatives"] == 1
        assert detector.get_metrics()["true_negatives"] == 0

    def test_false_negative_rate_calculation(self, detector):
        """Test calculation of false negative rate."""
        # Record multiple results
        for i in range(100):
            if i < 5:  # 5% false negatives
                # False negative
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_PATTERN
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=True,
                    accuracy=0.8
                )
            else:
                # True negative
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_SYNTAX
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=False,
                    accuracy=0.0
                )

            detector.record_validation_result(pruning_result, eval_result)

        metrics = detector.get_metrics()
        assert metrics["false_negatives"] == 5
        assert metrics["true_negatives"] == 95
        assert abs(metrics["false_negative_rate"] - 0.05) < 0.001

    def test_alert_generation(self, detector):
        """Test alert generation when threshold exceeded."""
        # Record enough samples to trigger alert
        for i in range(110):  # More than min_samples_for_alert
            if i < 10:  # ~9% false negatives (above 5% threshold)
                # False negative
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_CONFIDENCE
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=True,
                    accuracy=0.85
                )
            else:
                # True negative
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_PATTERN
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=False
                )

            detector.record_validation_result(pruning_result, eval_result)

        # Check alerts
        alerts = detector.get_active_alerts()
        assert len(alerts) > 0

        alert = alerts[0]
        assert isinstance(alert, FalseNegativeAlert)
        assert alert.false_negative_rate > detector.alert_threshold
        assert "exceeded threshold" in alert.message.lower()

    def test_windowed_rate_calculation(self, detector):
        """Test that only recent results are considered."""
        # Record old results (should be excluded)
        old_time = datetime.now() - timedelta(minutes=70)

        with patch('src.adapters.strategies.false_negative_detector.datetime') as mock_datetime:
            mock_datetime.now.return_value = old_time

            # Record 50 old false negatives
            for i in range(50):
                pruning_result = self.create_pruning_result(
                    f"old_prog{i}",
                    PruningDecision.REJECT_PATTERN
                )
                eval_result = self.create_evaluation_result(
                    f"old_prog{i}",
                    success=True,
                    accuracy=0.9
                )
                detector.record_validation_result(pruning_result, eval_result)

        # Record recent results
        for i in range(100):
            # All true negatives
            pruning_result = self.create_pruning_result(
                f"new_prog{i}",
                PruningDecision.REJECT_SYNTAX
            )
            eval_result = self.create_evaluation_result(
                f"new_prog{i}",
                success=False
            )
            detector.record_validation_result(pruning_result, eval_result)

        # Get windowed metrics
        windowed_metrics = detector.get_windowed_metrics(window_minutes=60)

        # Should only include recent results
        assert windowed_metrics["false_negatives"] == 0
        assert windowed_metrics["true_negatives"] == 100
        assert windowed_metrics["false_negative_rate"] == 0.0

    def test_strategy_specific_tracking(self, detector):
        """Test tracking false negatives by strategy."""
        strategies = ["conservative", "balanced", "aggressive"]

        # Record results for each strategy
        for strategy in strategies:
            for i in range(30):
                if strategy == "aggressive" and i < 5:
                    # More false negatives for aggressive strategy
                    pruning_result = self.create_pruning_result(
                        f"{strategy}_prog{i}",
                        PruningDecision.REJECT_CONFIDENCE
                    )
                    eval_result = self.create_evaluation_result(
                        f"{strategy}_prog{i}",
                        success=True,
                        accuracy=0.75
                    )
                else:
                    # True negative
                    pruning_result = self.create_pruning_result(
                        f"{strategy}_prog{i}",
                        PruningDecision.REJECT_PATTERN
                    )
                    eval_result = self.create_evaluation_result(
                        f"{strategy}_prog{i}",
                        success=False
                    )

                detector.record_validation_result(
                    pruning_result,
                    eval_result,
                    strategy_id=strategy
                )

        # Get strategy-specific metrics
        strategy_metrics = detector.get_strategy_metrics()

        assert "aggressive" in strategy_metrics
        assert strategy_metrics["aggressive"]["false_negatives"] == 5
        assert strategy_metrics["aggressive"]["false_negative_rate"] > 0.1

        assert strategy_metrics["conservative"]["false_negatives"] == 0
        assert strategy_metrics["balanced"]["false_negatives"] == 0

    def test_pattern_analysis(self, detector):
        """Test analysis of patterns in false negatives."""
        # Create false negatives with specific patterns
        rejection_reasons = [
            "Low confidence: 0.45",
            "Low confidence: 0.48",
            "Low confidence: 0.42",
            "Pattern rejection: contradiction",
            "Pattern rejection: contradiction",
        ]

        for i, reason in enumerate(rejection_reasons):
            pruning_result = PruningResult(
                program_id=f"fn_prog{i}",
                decision=PruningDecision.REJECT_CONFIDENCE if "confidence" in reason else PruningDecision.REJECT_PATTERN,
                confidence_score=0.45 if "confidence" in reason else 0.7,
                pruning_time_ms=5.0,
                rejection_reason=reason,
            )

            eval_result = self.create_evaluation_result(
                f"fn_prog{i}",
                success=True,
                accuracy=0.8
            )

            detector.record_validation_result(pruning_result, eval_result)

        # Analyze patterns
        patterns = detector.analyze_false_negative_patterns()

        assert "rejection_reasons" in patterns
        assert "Low confidence" in str(patterns["rejection_reasons"])
        assert patterns["most_common_decision"] == PruningDecision.REJECT_CONFIDENCE

    def test_alert_cooldown(self, detector):
        """Test that alerts have cooldown period."""
        # Trigger first alert
        for i in range(110):
            if i < 10:  # High false negative rate
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_PATTERN
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=True,
                    accuracy=0.9
                )
            else:
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_SYNTAX
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=False
                )

            detector.record_validation_result(pruning_result, eval_result)

        # First alert should be generated
        alerts1 = detector.get_active_alerts()
        assert len(alerts1) > 0

        # Try to trigger another alert immediately
        for i in range(110, 120):
            # More false negatives
            pruning_result = self.create_pruning_result(
                f"prog{i}",
                PruningDecision.REJECT_PATTERN
            )
            eval_result = self.create_evaluation_result(
                f"prog{i}",
                success=True,
                accuracy=0.9
            )
            detector.record_validation_result(pruning_result, eval_result)

        # Should not generate new alert due to cooldown
        alerts2 = detector.get_active_alerts()
        assert len(alerts2) == len(alerts1)  # Same alerts, no new ones

    def test_export_validation_report(self, detector):
        """Test export of validation report."""
        # Add some data
        for i in range(50):
            if i < 3:
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_CONFIDENCE
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=True,
                    accuracy=0.7
                )
            else:
                pruning_result = self.create_pruning_result(
                    f"prog{i}",
                    PruningDecision.REJECT_SYNTAX
                )
                eval_result = self.create_evaluation_result(
                    f"prog{i}",
                    success=False
                )

            detector.record_validation_result(pruning_result, eval_result)

        # Export report
        report = detector.export_validation_report()

        assert "summary" in report
        assert "metrics" in report
        assert "alerts" in report
        assert "patterns" in report
        assert "recommendations" in report

        assert report["metrics"]["total_validated"] == 50
        assert report["metrics"]["false_negatives"] == 3
