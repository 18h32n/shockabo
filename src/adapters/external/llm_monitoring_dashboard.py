"""LLM Monitoring Dashboard for tracking usage, costs, and performance."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.adapters.external.smart_model_router import SmartModelRouter
from src.infrastructure.components.budget_controller import BudgetController

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a specific model."""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency_seconds: float = 0.0
    complexity_distribution: dict[str, int] = field(default_factory=dict)
    hourly_usage: dict[str, int] = field(default_factory=dict)


class LLMMonitoringDashboard:
    """Dashboard for monitoring LLM usage and performance."""

    def __init__(
        self,
        budget_controller: BudgetController,
        model_router: SmartModelRouter,
        output_dir: Path = Path("reports/llm_monitoring")
    ):
        self.budget_controller = budget_controller
        self.model_router = model_router
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.routing_decisions: list[dict[str, Any]] = []

    def log_routing_decision(
        self,
        task_id: str,
        model_name: str,
        complexity_score: float,
        complexity_level: str,
        confidence: float,
        reasoning: str
    ):
        """Log a routing decision for analysis."""
        decision = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "model_name": model_name,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "confidence": confidence,
            "reasoning": reasoning
        }
        self.routing_decisions.append(decision)

    def log_request_performance(
        self,
        task_id: str,
        model_name: str,
        success: bool,
        latency_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ):
        """Log performance metrics for a request."""
        performance = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "model_name": model_name,
            "success": success,
            "latency_seconds": latency_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost
        }
        self.performance_history.append(performance)

    def generate_cost_report(self) -> dict[str, Any]:
        """Generate comprehensive cost analysis report."""
        budget_summary = self.budget_controller.get_usage_summary()

        # Calculate cost breakdown by model
        model_costs = budget_summary.get("model_usage", {})

        # Calculate cost trends
        hourly_costs = self._calculate_hourly_costs()
        daily_costs = self._calculate_daily_costs()

        # Budget projections
        current_rate = self._calculate_burn_rate()
        time_to_budget_limit = self._estimate_time_to_limit(current_rate)

        report = {
            "summary": {
                "total_spent": budget_summary["total_cost"],
                "budget_limit": budget_summary["budget_limit"],
                "remaining_budget": budget_summary["remaining_budget"],
                "utilization_percent": budget_summary["usage_percent"],
                "status": budget_summary["status"]
            },
            "model_breakdown": {
                model: {
                    "total_cost": data["cost"],
                    "request_count": data["count"],
                    "average_cost": data["cost"] / max(data["count"], 1),
                    "token_usage": {
                        "input": data["input_tokens"],
                        "output": data["output_tokens"],
                        "total": data["input_tokens"] + data["output_tokens"]
                    }
                }
                for model, data in model_costs.items()
            },
            "trends": {
                "hourly_costs": hourly_costs,
                "daily_costs": daily_costs,
                "burn_rate_per_hour": current_rate,
                "estimated_hours_remaining": time_to_budget_limit
            },
            "recommendations": self._generate_cost_recommendations(budget_summary)
        }

        return report

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate model performance analysis report."""
        performance_summary = self.model_router.get_performance_summary()

        # Aggregate metrics by model
        model_metrics = self._aggregate_model_metrics()

        # Calculate success rates (used for internal analysis)
        _ = self._calculate_success_rates()

        # Analyze routing efficiency
        routing_efficiency = self._analyze_routing_efficiency()

        report = {
            "model_performance": {
                model: {
                    "total_requests": metrics.total_requests,
                    "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
                    "average_latency": metrics.average_latency_seconds,
                    "total_tokens": metrics.total_tokens,
                    "complexity_distribution": metrics.complexity_distribution,
                    "cost_efficiency": metrics.total_tokens / max(metrics.total_cost, 0.01)
                }
                for model, metrics in model_metrics.items()
            },
            "routing_analysis": {
                "total_routing_decisions": len(self.routing_decisions),
                "complexity_distribution": self._get_complexity_distribution(),
                "confidence_metrics": self._calculate_confidence_metrics(),
                "routing_efficiency": routing_efficiency
            },
            "circuit_breaker_status": performance_summary.get("circuit_breaker_status", {}),
            "recommendations": self._generate_performance_recommendations(model_metrics)
        }

        return report

    def visualize_cost_trends(self, save_path: Path | None = None):
        """Create visualization of cost trends."""
        if not self.performance_history:
            logger.warning("No performance history to visualize")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('LLM Cost and Usage Trends', fontsize=16)

        # 1. Cumulative cost over time
        ax1 = axes[0, 0]
        df['cumulative_cost'] = df['cost'].cumsum()
        df['cumulative_cost'].plot(ax=ax1, color='red', linewidth=2)
        ax1.axhline(y=self.budget_controller.budget_limit, color='red',
                    linestyle='--', label='Budget Limit')
        ax1.set_title('Cumulative Cost Over Time')
        ax1.set_ylabel('Cost ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cost by model
        ax2 = axes[0, 1]
        model_costs = df.groupby('model_name')['cost'].sum().sort_values(ascending=True)
        model_costs.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title('Total Cost by Model')
        ax2.set_xlabel('Cost ($)')

        # 3. Hourly cost rate
        ax3 = axes[1, 0]
        hourly_costs = df.resample('H')['cost'].sum()
        hourly_costs.plot(ax=ax3, kind='bar', color='green', alpha=0.7)
        ax3.set_title('Hourly Cost Rate')
        ax3.set_ylabel('Cost ($)')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Token usage efficiency
        ax4 = axes[1, 1]
        df['tokens_per_dollar'] = (df['input_tokens'] + df['output_tokens']) / df['cost'].clip(lower=0.001)
        model_efficiency = df.groupby('model_name')['tokens_per_dollar'].mean().sort_values()
        model_efficiency.plot(kind='barh', ax=ax4, color='orange')
        ax4.set_title('Token Efficiency by Model')
        ax4.set_xlabel('Tokens per Dollar')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"cost_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()
        logger.info(f"Cost trends visualization saved to {save_path}")

    def visualize_routing_decisions(self, save_path: Path | None = None):
        """Visualize routing decision patterns."""
        if not self.routing_decisions:
            logger.warning("No routing decisions to visualize")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.routing_decisions)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Routing Analysis', fontsize=16)

        # 1. Complexity score distribution
        ax1 = axes[0, 0]
        df['complexity_score'].hist(bins=20, ax=ax1, color='purple', alpha=0.7, edgecolor='black')
        ax1.set_title('Task Complexity Distribution')
        ax1.set_xlabel('Complexity Score')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)

        # 2. Model selection frequency
        ax2 = axes[0, 1]
        model_counts = df['model_name'].value_counts()
        model_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Model Selection Distribution')
        ax2.set_ylabel('')

        # 3. Confidence by complexity level
        ax3 = axes[1, 0]
        complexity_confidence = df.groupby('complexity_level')['confidence'].mean().sort_values()
        complexity_confidence.plot(kind='bar', ax=ax3, color='coral')
        ax3.set_title('Average Routing Confidence by Complexity')
        ax3.set_ylabel('Confidence Score')
        ax3.set_xlabel('Complexity Level')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Complexity vs Model scatter
        ax4 = axes[1, 1]
        models = df['model_name'].unique()
        colors = plt.cm.tab10(range(len(models)))
        for i, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            ax4.scatter(model_df['complexity_score'], model_df['confidence'],
                       label=model, alpha=0.6, color=colors[i])
        ax4.set_title('Routing Decisions: Complexity vs Confidence')
        ax4.set_xlabel('Complexity Score')
        ax4.set_ylabel('Confidence')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"routing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()
        logger.info(f"Routing analysis visualization saved to {save_path}")

    def export_detailed_report(self, format: str = "json") -> Path:
        """Export comprehensive monitoring report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate all reports
        cost_report = self.generate_cost_report()
        performance_report = self.generate_performance_report()

        # Combine into comprehensive report
        full_report = {
            "generated_at": datetime.now().isoformat(),
            "cost_analysis": cost_report,
            "performance_analysis": performance_report,
            "routing_history": self.routing_decisions[-100:],  # Last 100 decisions
            "performance_history": self.performance_history[-100:]  # Last 100 requests
        }

        # Export based on format
        if format == "json":
            output_path = self.output_dir / f"llm_report_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(full_report, f, indent=2)

        elif format == "html":
            output_path = self.output_dir / f"llm_report_{timestamp}.html"
            html_content = self._generate_html_report(full_report)
            with open(output_path, 'w') as f:
                f.write(html_content)

        else:
            raise ValueError(f"Unsupported format: {format}")

        # Also generate visualizations
        self.visualize_cost_trends()
        self.visualize_routing_decisions()

        logger.info(f"Detailed report exported to {output_path}")
        return output_path

    # Private helper methods
    def _calculate_hourly_costs(self) -> dict[str, float]:
        """Calculate costs per hour for the last 24 hours."""
        if not self.performance_history:
            return {}

        now = datetime.now()
        hourly_costs = {}

        for i in range(24):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)

            hour_cost = sum(
                perf['cost']
                for perf in self.performance_history
                if hour_start <= datetime.fromisoformat(perf['timestamp']) < hour_end
            )

            hourly_costs[hour_start.strftime('%Y-%m-%d %H:00')] = hour_cost

        return hourly_costs

    def _calculate_daily_costs(self) -> dict[str, float]:
        """Calculate costs per day for the last 7 days."""
        if not self.performance_history:
            return {}

        now = datetime.now()
        daily_costs = {}

        for i in range(7):
            day_start = (now - timedelta(days=i+1)).replace(hour=0, minute=0, second=0)
            day_end = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0)

            day_cost = sum(
                perf['cost']
                for perf in self.performance_history
                if day_start <= datetime.fromisoformat(perf['timestamp']) < day_end
            )

            daily_costs[day_start.strftime('%Y-%m-%d')] = day_cost

        return daily_costs

    def _calculate_burn_rate(self) -> float:
        """Calculate current burn rate per hour."""
        recent_costs = self._calculate_hourly_costs()
        if not recent_costs:
            return 0.0

        # Average of last 3 hours
        recent_hours = sorted(recent_costs.keys(), reverse=True)[:3]
        if not recent_hours:
            return 0.0

        return sum(recent_costs[hour] for hour in recent_hours) / len(recent_hours)

    def _estimate_time_to_limit(self, burn_rate: float) -> float:
        """Estimate hours until budget limit reached."""
        if burn_rate <= 0:
            return float('inf')

        remaining = self.budget_controller.get_remaining_budget()
        return remaining / burn_rate

    def _aggregate_model_metrics(self) -> dict[str, ModelPerformanceMetrics]:
        """Aggregate performance metrics by model."""
        metrics = {}

        for perf in self.performance_history:
            model = perf['model_name']
            if model not in metrics:
                metrics[model] = ModelPerformanceMetrics(model_name=model)

            m = metrics[model]
            m.total_requests += 1
            if perf['success']:
                m.successful_requests += 1
            else:
                m.failed_requests += 1

            m.total_tokens += perf['input_tokens'] + perf['output_tokens']
            m.total_cost += perf['cost']

            # Update average latency
            m.average_latency_seconds = (
                (m.average_latency_seconds * (m.total_requests - 1) + perf['latency_seconds'])
                / m.total_requests
            )

        # Add complexity distribution from routing decisions
        for decision in self.routing_decisions:
            model = decision['model_name']
            if model in metrics:
                level = decision['complexity_level']
                metrics[model].complexity_distribution[level] = (
                    metrics[model].complexity_distribution.get(level, 0) + 1
                )

        return metrics

    def _calculate_success_rates(self) -> dict[str, float]:
        """Calculate success rates by model."""
        model_success = {}
        model_total = {}

        for perf in self.performance_history:
            model = perf['model_name']
            model_total[model] = model_total.get(model, 0) + 1
            if perf['success']:
                model_success[model] = model_success.get(model, 0) + 1

        return {
            model: model_success.get(model, 0) / total
            for model, total in model_total.items()
        }

    def _analyze_routing_efficiency(self) -> dict[str, Any]:
        """Analyze efficiency of routing decisions."""
        if not self.routing_decisions:
            return {}

        # Group by complexity level
        complexity_groups = {}
        for decision in self.routing_decisions:
            level = decision['complexity_level']
            if level not in complexity_groups:
                complexity_groups[level] = []
            complexity_groups[level].append(decision)

        # Analyze each group
        efficiency_metrics = {}
        for level, decisions in complexity_groups.items():
            avg_confidence = sum(d['confidence'] for d in decisions) / len(decisions)
            model_diversity = len({d['model_name'] for d in decisions})

            efficiency_metrics[level] = {
                "count": len(decisions),
                "average_confidence": avg_confidence,
                "model_diversity": model_diversity,
                "primary_model": max(
                    {d['model_name'] for d in decisions},
                    key=lambda m: sum(1 for d in decisions if d['model_name'] == m)
                )
            }

        return efficiency_metrics

    def _get_complexity_distribution(self) -> dict[str, int]:
        """Get distribution of complexity levels."""
        distribution = {}
        for decision in self.routing_decisions:
            level = decision['complexity_level']
            distribution[level] = distribution.get(level, 0) + 1
        return distribution

    def _calculate_confidence_metrics(self) -> dict[str, float]:
        """Calculate confidence metrics for routing."""
        if not self.routing_decisions:
            return {}

        confidences = [d['confidence'] for d in self.routing_decisions]
        return {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "low_confidence_ratio": sum(1 for c in confidences if c < 0.5) / len(confidences)
        }

    def _generate_cost_recommendations(self, budget_summary: dict[str, Any]) -> list[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        usage_percent = budget_summary.get("usage_percent", 0)

        if usage_percent > 80:
            recommendations.append("⚠️ Budget usage critical - consider disabling high-cost models")
        elif usage_percent > 50:
            recommendations.append("📊 Budget usage over 50% - monitor closely")

        # Check model costs
        model_usage = budget_summary.get("model_usage", {})
        for model, data in model_usage.items():
            avg_cost = data["cost"] / max(data["count"], 1)
            if avg_cost > 1.0:
                recommendations.append(f"💰 {model} has high average cost (${avg_cost:.2f}/request)")

        # Check burn rate
        burn_rate = self._calculate_burn_rate()
        if burn_rate > 5.0:
            recommendations.append(f"🔥 High burn rate: ${burn_rate:.2f}/hour")

        return recommendations

    def _generate_performance_recommendations(
        self,
        model_metrics: dict[str, ModelPerformanceMetrics]
    ) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        for model, metrics in model_metrics.items():
            # Check success rate
            success_rate = metrics.successful_requests / max(metrics.total_requests, 1)
            if success_rate < 0.5:
                recommendations.append(f"❌ {model} has low success rate ({success_rate:.1%})")

            # Check latency
            if metrics.average_latency_seconds > 30:
                recommendations.append(
                    f"⏱️ {model} has high latency ({metrics.average_latency_seconds:.1f}s avg)"
                )

            # Check cost efficiency
            if metrics.total_cost > 0:
                tokens_per_dollar = metrics.total_tokens / metrics.total_cost
                if tokens_per_dollar < 1000:
                    recommendations.append(
                        f"📉 {model} has low token efficiency ({tokens_per_dollar:.0f} tokens/$)"
                    )

        return recommendations

    def _generate_html_report(self, report_data: dict[str, Any]) -> str:
        """Generate HTML report from data."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .warning {{ color: #ff9800; }}
        .critical {{ color: #f44336; }}
        .success {{ color: #4caf50; }}
    </style>
</head>
<body>
    <h1>LLM Monitoring Report</h1>
    <p>Generated: {report_data['generated_at']}</p>

    <h2>Cost Summary</h2>
    <div>
        <span class="metric">${report_data['cost_analysis']['summary']['total_spent']:.2f}</span>
        of ${report_data['cost_analysis']['summary']['budget_limit']:.2f} budget used
        ({report_data['cost_analysis']['summary']['utilization_percent']:.1f}%)
    </div>

    <h2>Model Performance</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Requests</th>
            <th>Success Rate</th>
            <th>Avg Latency</th>
            <th>Total Cost</th>
        </tr>
        {self._generate_performance_table_rows(report_data['performance_analysis']['model_performance'])}
    </table>

    <h2>Recommendations</h2>
    <ul>
        {self._generate_recommendation_list(report_data)}
    </ul>
</body>
</html>
        """
        return html

    def _generate_performance_table_rows(self, model_performance: dict[str, Any]) -> str:
        """Generate HTML table rows for model performance."""
        rows = []
        for model, perf in model_performance.items():
            success_class = (
                "success" if perf['success_rate'] > 0.8
                else "warning" if perf['success_rate'] > 0.5
                else "critical"
            )
            rows.append(f"""
                <tr>
                    <td>{model}</td>
                    <td>{perf['total_requests']}</td>
                    <td class="{success_class}">{perf['success_rate']:.1%}</td>
                    <td>{perf['average_latency']:.1f}s</td>
                    <td>${perf.get('total_cost', 0):.2f}</td>
                </tr>
            """)
        return "\n".join(rows)

    def _generate_recommendation_list(self, report_data: dict[str, Any]) -> str:
        """Generate HTML list of recommendations."""
        all_recommendations = (
            report_data['cost_analysis'].get('recommendations', []) +
            report_data['performance_analysis'].get('recommendations', [])
        )

        if not all_recommendations:
            return "<li>No recommendations at this time.</li>"

        return "\n".join(f"<li>{rec}</li>" for rec in all_recommendations)
