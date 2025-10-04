"""Performance analytics dashboard for program cache."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    pass



@dataclass
class TimeSeriesMetric:
    """Time series metric data point."""
    timestamp: datetime
    value: float
    label: str | None = None


@dataclass
class AnalyticsDashboard:
    """Dashboard for program cache analytics."""

    cache: Any  # ProgramCache
    output_dir: Path = field(default_factory=lambda: Path("analytics"))

    def __post_init__(self):
        """Initialize dashboard."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics_history.json"
        self.metrics_history = self._load_metrics_history()

    def collect_metrics(self) -> dict[str, Any]:
        """Collect current metrics from cache."""
        stats = self.cache.get_statistics()
        patterns = self.cache.get_patterns()

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': {
                'total_programs': stats.total_programs,
                'successful_programs': stats.successful_programs,
                'unique_programs': stats.unique_programs,
                'total_size_mb': stats.total_size_bytes / (1024 * 1024),
                'cache_hit_rate': stats.cache_hit_rate,
                'average_access_frequency': stats.average_access_frequency
            },
            'task_distribution': stats.task_type_distribution,
            'generation_distribution': stats.generation_distribution,
            'pattern_count': len(patterns),
            'top_patterns': stats.most_successful_patterns[:5]
        }

        # Add to history
        self.metrics_history.append(metrics)
        self._save_metrics_history()

        return metrics

    def generate_analytics_report(
        self,
        format: str = 'html',
        include_visuals: bool = True
    ) -> str:
        """Generate comprehensive analytics report."""
        # Collect current metrics
        current_metrics = self.collect_metrics()

        # Generate visualizations
        if include_visuals:
            figures = self._create_visualizations()

        # Create report
        if format == 'html':
            return self._generate_html_report(current_metrics, figures if include_visuals else [])
        elif format == 'pdf':
            return self._generate_pdf_report(current_metrics, figures if include_visuals else [])
        elif format == 'json':
            return self._generate_json_report(current_metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _create_visualizations(self) -> list[Any]:
        """Create all dashboard visualizations."""
        figures = []

        # 1. Cache metrics over time
        fig_cache = self._plot_cache_metrics_timeline()
        if fig_cache:
            figures.append(('cache_timeline', fig_cache))

        # 2. Success rate by pattern
        fig_patterns = self._plot_pattern_success_rates()
        if fig_patterns:
            figures.append(('pattern_success', fig_patterns))

        # 3. Task type distribution
        fig_tasks = self._plot_task_distribution()
        if fig_tasks:
            figures.append(('task_distribution', fig_tasks))

        # 4. Generation distribution (for evolution)
        fig_gens = self._plot_generation_distribution()
        if fig_gens:
            figures.append(('generation_dist', fig_gens))

        # 5. Cache efficiency heatmap
        fig_efficiency = self._plot_cache_efficiency()
        if fig_efficiency:
            figures.append(('cache_efficiency', fig_efficiency))

        return figures

    def _plot_cache_metrics_timeline(self) -> Any | None:
        """Plot cache metrics over time."""
        if not self.metrics_history or len(self.metrics_history) < 2:
            return None

        # Extract time series data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metrics_history]
        hit_rates = [m['cache_stats']['cache_hit_rate'] for m in self.metrics_history]
        program_counts = [m['cache_stats']['total_programs'] for m in self.metrics_history]
        sizes_mb = [m['cache_stats']['total_size_mb'] for m in self.metrics_history]

        if PLOTLY_AVAILABLE:
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Cache Hit Rate', 'Total Programs', 'Cache Size (MB)'),
                shared_xaxes=True
            )

            # Hit rate
            fig.add_trace(
                go.Scatter(x=timestamps, y=hit_rates, mode='lines+markers', name='Hit Rate'),
                row=1, col=1
            )

            # Program count
            fig.add_trace(
                go.Scatter(x=timestamps, y=program_counts, mode='lines+markers', name='Programs'),
                row=2, col=1
            )

            # Cache size
            fig.add_trace(
                go.Scatter(x=timestamps, y=sizes_mb, mode='lines+markers', name='Size (MB)'),
                row=3, col=1
            )

            fig.update_layout(height=800, title='Cache Metrics Over Time')
            return fig
        else:
            # Matplotlib fallback
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            axes[0].plot(timestamps, hit_rates, 'b-o')
            axes[0].set_ylabel('Cache Hit Rate')
            axes[0].set_title('Cache Hit Rate Over Time')
            axes[0].grid(True)

            axes[1].plot(timestamps, program_counts, 'g-o')
            axes[1].set_ylabel('Total Programs')
            axes[1].set_title('Total Programs Over Time')
            axes[1].grid(True)

            axes[2].plot(timestamps, sizes_mb, 'r-o')
            axes[2].set_ylabel('Cache Size (MB)')
            axes[2].set_title('Cache Size Over Time')
            axes[2].grid(True)

            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45)
            plt.tight_layout()

            return fig

    def _plot_pattern_success_rates(self) -> Any | None:
        """Plot success rates by pattern."""
        patterns = self.cache.get_patterns()
        if not patterns:
            return None

        # Get top patterns by frequency * success rate
        sorted_patterns = sorted(
            patterns.values(),
            key=lambda p: p.frequency * p.success_rate,
            reverse=True
        )[:10]  # Top 10

        pattern_names = [p.pattern_description[:50] + '...' if len(p.pattern_description) > 50
                        else p.pattern_description for p in sorted_patterns]
        success_rates = [p.success_rate for p in sorted_patterns]
        frequencies = [p.frequency for p in sorted_patterns]

        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            # Success rate bars
            fig.add_trace(go.Bar(
                name='Success Rate',
                x=pattern_names,
                y=success_rates,
                yaxis='y',
                marker_color='lightblue'
            ))

            # Frequency line
            fig.add_trace(go.Scatter(
                name='Frequency',
                x=pattern_names,
                y=frequencies,
                yaxis='y2',
                line={"color": 'red', "width": 2},
                marker={"size": 8}
            ))

            # Update layout
            fig.update_layout(
                title='Pattern Success Rates and Frequencies',
                xaxis_tickangle=-45,
                yaxis={"title": 'Success Rate', "side": 'left'},
                yaxis2={"title": 'Frequency', "overlaying": 'y', "side": 'right'},
                height=600
            )

            return fig
        else:
            # Matplotlib
            fig, ax1 = plt.subplots(figsize=(12, 6))

            x = np.arange(len(pattern_names))
            width = 0.8

            # Success rate bars
            ax1.bar(x, success_rates, width, label='Success Rate', color='lightblue')
            ax1.set_ylabel('Success Rate', color='blue')
            ax1.set_ylim(0, 1.1)
            ax1.tick_params(axis='y', labelcolor='blue')

            # Frequency line
            ax2 = ax1.twinx()
            ax2.plot(x, frequencies, 'r-o', linewidth=2, markersize=8, label='Frequency')
            ax2.set_ylabel('Frequency', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Labels
            ax1.set_xlabel('Pattern')
            ax1.set_title('Pattern Success Rates and Frequencies')
            ax1.set_xticks(x)
            ax1.set_xticklabels(pattern_names, rotation=45, ha='right')

            # Legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.tight_layout()
            return fig

    def _plot_task_distribution(self) -> Any | None:
        """Plot task type distribution."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        if not current_metrics or not current_metrics.get('task_distribution'):
            return None

        task_dist = current_metrics['task_distribution']

        if PLOTLY_AVAILABLE:
            fig = px.pie(
                values=list(task_dist.values()),
                names=list(task_dist.keys()),
                title='Task Type Distribution'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                task_dist.values(),
                labels=task_dist.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Task Type Distribution')
            return fig

    def _plot_generation_distribution(self) -> Any | None:
        """Plot generation distribution for evolution programs."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        if not current_metrics or not current_metrics.get('generation_distribution'):
            return None

        gen_dist = current_metrics['generation_distribution']
        if not gen_dist:
            return None

        generations = sorted([int(g) for g in gen_dist.keys()])
        counts = [gen_dist[str(g)] for g in generations]

        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(x=generations, y=counts)
            ])
            fig.update_layout(
                title='Evolution Generation Distribution',
                xaxis_title='Generation',
                yaxis_title='Program Count'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(generations, counts)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Program Count')
            ax.set_title('Evolution Generation Distribution')
            ax.grid(True, axis='y')
            return fig

    def _plot_cache_efficiency(self) -> Any | None:
        """Plot cache efficiency metrics."""
        if len(self.metrics_history) < 24:  # Need at least 24 hours of data
            return None

        # Create hourly efficiency matrix
        hourly_data = []
        for metric in self.metrics_history[-168:]:  # Last week
            timestamp = datetime.fromisoformat(metric['timestamp'])
            hour = timestamp.hour
            day = timestamp.weekday()
            hit_rate = metric['cache_stats']['cache_hit_rate']
            hourly_data.append((day, hour, hit_rate))

        # Create matrix
        efficiency_matrix = np.zeros((7, 24))
        counts = np.zeros((7, 24))

        for day, hour, rate in hourly_data:
            efficiency_matrix[day, hour] += rate
            counts[day, hour] += 1

        # Average
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency_matrix = np.divide(efficiency_matrix, counts)
            efficiency_matrix[np.isnan(efficiency_matrix)] = 0

        if PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=efficiency_matrix,
                x=list(range(24)),
                y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                colorscale='Viridis'
            ))
            fig.update_layout(
                title='Cache Hit Rate by Day and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week'
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(efficiency_matrix, cmap='viridis', aspect='auto')

            # Labels
            ax.set_xticks(np.arange(24))
            ax.set_xticklabels(range(24))
            ax.set_yticks(np.arange(7))
            ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

            # Colorbar
            plt.colorbar(im, ax=ax, label='Cache Hit Rate')

            ax.set_title('Cache Hit Rate by Day and Hour')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Day of Week')

            return fig

    def _generate_html_report(self, metrics: dict[str, Any], figures: list[tuple[str, Any]]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Program Cache Analytics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background: #f0f0f0;
                    border-radius: 5px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
                .metric-label {{ color: #666; }}
                .pattern-list {{ margin: 20px 0; }}
                .pattern-item {{ margin: 5px 0; padding: 5px; background: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Program Cache Analytics Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Current Metrics</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Total Programs</div>
                    <div class="metric-value">{metrics['cache_stats']['total_programs']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{metrics['cache_stats']['successful_programs'] / max(1, metrics['cache_stats']['total_programs']) * 100:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Cache Hit Rate</div>
                    <div class="metric-value">{metrics['cache_stats']['cache_hit_rate'] * 100:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Cache Size</div>
                    <div class="metric-value">{metrics['cache_stats']['total_size_mb']:.1f} MB</div>
                </div>
            </div>

            <h2>Top Patterns</h2>
            <div class="pattern-list">
        """

        for pattern in metrics.get('top_patterns', []):
            html += f'<div class="pattern-item">{pattern}</div>\n'

        html += """
            </div>
        """

        # Add figures if using plotly
        if PLOTLY_AVAILABLE:
            for name, fig in figures:
                html += f"<h2>{name.replace('_', ' ').title()}</h2>\n"
                html += fig.to_html(include_plotlyjs='cdn')
                html += "\n"

        html += """
        </body>
        </html>
        """

        # Save report
        report_path = self.output_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_path, 'w') as f:
            f.write(html)

        return str(report_path)

    def _generate_pdf_report(self, metrics: dict[str, Any], figures: list[tuple[str, Any]]) -> str:
        """Generate PDF report."""
        report_path = self.output_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        with PdfPages(report_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.9, 'Program Cache Analytics Report',
                    ha='center', size=24, weight='bold')
            fig.text(0.5, 0.8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ha='center', size=12)

            # Metrics summary
            y_pos = 0.7
            for key, value in metrics['cache_stats'].items():
                fig.text(0.2, y_pos, f"{key.replace('_', ' ').title()}:", size=12)
                fig.text(0.6, y_pos, f"{value:.2f}" if isinstance(value, float) else str(value), size=12)
                y_pos -= 0.05

            plt.axis('off')
            pdf.savefig(fig)
            plt.close()

            # Add visualization figures
            for _name, fig in figures:
                if isinstance(fig, Figure):  # Matplotlib figure
                    pdf.savefig(fig)
                    plt.close(fig)

        return str(report_path)

    def _generate_json_report(self, metrics: dict[str, Any]) -> str:
        """Generate JSON report."""
        report_path = self.output_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        return str(report_path)

    def export_metrics_csv(self, output_file: str | None = None) -> str:
        """Export metrics history to CSV."""
        if not self.metrics_history:
            return ""

        if output_file is None:
            output_file = self.output_dir / f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Flatten metrics for CSV
        rows = []
        for metric in self.metrics_history:
            row = {
                'timestamp': metric['timestamp'],
                'total_programs': metric['cache_stats']['total_programs'],
                'successful_programs': metric['cache_stats']['successful_programs'],
                'cache_hit_rate': metric['cache_stats']['cache_hit_rate'],
                'cache_size_mb': metric['cache_stats']['total_size_mb'],
                'pattern_count': metric.get('pattern_count', 0)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        return str(output_file)

    def _load_metrics_history(self) -> list[dict[str, Any]]:
        """Load metrics history from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_metrics_history(self) -> None:
        """Save metrics history to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metrics history: {e}")
