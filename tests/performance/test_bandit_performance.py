"""Performance tests for Multi-Armed Bandit Controller.

Validates 20%+ improvement over fixed allocation using A/B testing.
"""

import time

import numpy as np
import pytest
from scipy import stats

from src.adapters.strategies.bandit_controller import BanditController


class BaselineController:
    """Fixed allocation baseline for comparison."""

    def __init__(self, strategies: list[str]):
        self.strategies = strategies
        self.selection_count = 0

    def select_strategy(self, task_features: dict[str, float] = None) -> str:
        """Uniform random selection."""
        strategy = self.strategies[self.selection_count % len(self.strategies)]
        self.selection_count += 1
        return strategy

    def update_reward(self, strategy_id: str, reward: float, cost: float = 0.0) -> None:
        """No-op for baseline."""
        pass


class MockEvolutionScenario:
    """Simulates evolution scenarios with known strategy performance profiles."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.strategies = [
            "hybrid_init",
            "pure_llm",
            "dsl_mutation",
            "crossover_focused",
            "adaptive_mutation",
        ]

        self.strategy_performance = {
            "hybrid_init": {"mean": 0.55, "std": 0.18, "cost": 0.7},
            "pure_llm": {"mean": 0.70, "std": 0.15, "cost": 1.0},
            "dsl_mutation": {"mean": 0.35, "std": 0.22, "cost": 0.1},
            "crossover_focused": {"mean": 0.30, "std": 0.20, "cost": 0.1},
            "adaptive_mutation": {"mean": 0.40, "std": 0.19, "cost": 0.15},
        }

    def generate_task_features(self) -> dict[str, float]:
        """Generate random task features."""
        return {
            "grid_size": self.rng.uniform(0.1, 1.0),
            "color_diversity": self.rng.uniform(0.2, 1.0),
            "edge_density": self.rng.uniform(0.1, 0.8),
            "symmetry_score": self.rng.uniform(0.0, 1.0),
        }

    def simulate_strategy_execution(
        self, strategy_id: str, task_features: dict[str, float]
    ) -> tuple[float, float]:
        """Simulate strategy execution with contextual variation."""
        profile = self.strategy_performance[strategy_id]

        context_bonus = 0.0
        if strategy_id == "hybrid_init" and task_features["color_diversity"] > 0.7:
            context_bonus = 0.25
        elif strategy_id == "pure_llm" and task_features["grid_size"] < 0.4:
            context_bonus = 0.20
        elif strategy_id == "dsl_mutation" and task_features["edge_density"] > 0.6:
            context_bonus = 0.35
        elif strategy_id == "crossover_focused" and task_features["symmetry_score"] > 0.7:
            context_bonus = 0.30
        elif strategy_id == "adaptive_mutation" and task_features["grid_size"] > 0.6:
            context_bonus = 0.25

        fitness = max(
            0.0, min(1.0, self.rng.normal(profile["mean"] + context_bonus, profile["std"]))
        )

        cost = profile["cost"] * self.rng.uniform(0.8, 1.2)

        return fitness, cost


@pytest.fixture
def evolution_scenario():
    """Create mock evolution scenario."""
    return MockEvolutionScenario(seed=42)


@pytest.fixture
def strategies():
    """Strategy list for testing."""
    return [
        "hybrid_init",
        "pure_llm",
        "dsl_mutation",
        "crossover_focused",
        "adaptive_mutation",
    ]


def run_experiment(
    controller,
    scenario: MockEvolutionScenario,
    num_tasks: int = 100,
    num_generations: int = 10,
    cost_aware: bool = False,
) -> dict[str, float]:
    """Run evolution experiment with given controller."""
    total_fitness = 0.0
    total_cost = 0.0
    convergence_times = []
    generation_count = 0

    for task_idx in range(num_tasks):
        task_features = scenario.generate_task_features()
        task_best_fitness = 0.0
        task_converged = False

        for gen in range(num_generations):
            generation_count += 1
            strategy = controller.select_strategy(task_features)
            fitness, cost = scenario.simulate_strategy_execution(strategy, task_features)

            total_fitness += fitness
            total_cost += cost

            if cost_aware:
                cost_adjusted_reward = fitness / (1 + 0.2 * cost)
                controller.update_reward(strategy, cost_adjusted_reward, cost)
            else:
                controller.update_reward(strategy, fitness, cost)

            if fitness > task_best_fitness:
                task_best_fitness = fitness

            if fitness > 0.95 and not task_converged:
                convergence_times.append(gen + 1)
                task_converged = True

    avg_fitness = total_fitness / generation_count
    avg_cost = total_cost / generation_count
    avg_convergence = (
        sum(convergence_times) / len(convergence_times) if convergence_times else num_generations
    )
    cost_efficiency = avg_fitness / (avg_cost + 0.01)

    return {
        "avg_fitness": avg_fitness,
        "avg_cost": avg_cost,
        "avg_convergence": avg_convergence,
        "cost_efficiency": cost_efficiency,
        "convergence_rate": len(convergence_times) / num_tasks,
    }


@pytest.mark.performance
def test_bandit_vs_baseline_fitness(evolution_scenario, strategies):
    """Test that MAB achieves 20%+ fitness improvement over baseline."""
    baseline = BaselineController(strategies)
    bandit = BanditController(
        strategies=strategies, alpha_prior=1.0, beta_prior=1.0, warmup_selections=20
    )

    baseline_results = run_experiment(baseline, evolution_scenario, num_tasks=100)
    bandit_results = run_experiment(bandit, evolution_scenario, num_tasks=100)

    fitness_improvement = (
        (bandit_results["avg_fitness"] - baseline_results["avg_fitness"])
        / baseline_results["avg_fitness"]
        * 100
    )

    assert fitness_improvement >= 20.0, (
        f"Fitness improvement {fitness_improvement:.1f}% < 20% threshold. "
        f"Baseline: {baseline_results['avg_fitness']:.3f}, "
        f"Bandit: {bandit_results['avg_fitness']:.3f}"
    )


@pytest.mark.performance
def test_bandit_vs_baseline_convergence(evolution_scenario, strategies):
    """Test that MAB achieves 15%+ convergence speed improvement."""
    baseline = BaselineController(strategies)
    bandit = BanditController(
        strategies=strategies, alpha_prior=1.0, beta_prior=1.0, warmup_selections=20
    )

    baseline_results = run_experiment(baseline, evolution_scenario, num_tasks=100)
    bandit_results = run_experiment(bandit, evolution_scenario, num_tasks=100)

    convergence_improvement = (
        (baseline_results["avg_convergence"] - bandit_results["avg_convergence"])
        / baseline_results["avg_convergence"]
        * 100
    )

    assert convergence_improvement >= 15.0, (
        f"Convergence improvement {convergence_improvement:.1f}% < 15% threshold. "
        f"Baseline: {baseline_results['avg_convergence']:.1f} gens, "
        f"Bandit: {bandit_results['avg_convergence']:.1f} gens"
    )


@pytest.mark.performance
def test_bandit_with_cost_aware_rewards(evolution_scenario, strategies):
    """Test that MAB with cost-aware rewards successfully balances fitness and cost."""
    bandit = BanditController(
        strategies=strategies,
        alpha_prior=1.0,
        beta_prior=1.0,
        warmup_selections=50,
        success_threshold=0.35,
    )

    bandit_results = run_experiment(bandit, evolution_scenario, num_tasks=100, cost_aware=True)

    assert (
        bandit_results["avg_fitness"] >= 0.50
    ), f"Cost-aware bandit fitness {bandit_results['avg_fitness']:.3f} < 0.50 threshold"

    assert (
        bandit_results["cost_efficiency"] >= 0.75
    ), f"Cost-aware bandit efficiency {bandit_results['cost_efficiency']:.3f} < 0.75 threshold"

    assert (
        bandit_results["avg_cost"] <= 1.0
    ), f"Cost-aware bandit cost {bandit_results['avg_cost']:.3f} > 1.0 threshold"


@pytest.mark.performance
def test_statistical_significance(evolution_scenario, strategies):
    """Test that improvements are statistically significant (p < 0.05)."""
    num_runs = 10
    baseline_fitness = []
    bandit_fitness = []

    for run in range(num_runs):
        scenario = MockEvolutionScenario(seed=42 + run)

        baseline = BaselineController(strategies)
        baseline_results = run_experiment(baseline, scenario, num_tasks=20)
        baseline_fitness.append(baseline_results["avg_fitness"])

        bandit = BanditController(
            strategies=strategies,
            alpha_prior=1.0,
            beta_prior=1.0,
            warmup_selections=20,
            success_threshold=0.4,
        )
        bandit_results = run_experiment(bandit, scenario, num_tasks=20)
        bandit_fitness.append(bandit_results["avg_fitness"])

    t_stat, p_value = stats.ttest_ind(bandit_fitness, baseline_fitness)

    assert p_value < 0.05, (
        f"Improvement not statistically significant (p={p_value:.4f} >= 0.05). "
        f"Baseline mean: {np.mean(baseline_fitness):.3f}, "
        f"Bandit mean: {np.mean(bandit_fitness):.3f}"
    )

    assert np.mean(bandit_fitness) > np.mean(
        baseline_fitness
    ), "Bandit fitness should be higher than baseline"


@pytest.mark.performance
def test_selection_latency(strategies):
    """Test that strategy selection meets <10ms latency requirement."""
    bandit = BanditController(
        strategies=strategies, alpha_prior=1.0, beta_prior=1.0, warmup_selections=20
    )

    task_features = {"grid_size": 0.5, "color_diversity": 0.7, "edge_density": 0.4}

    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        strategy = bandit.select_strategy(task_features)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    assert avg_latency < 10.0, f"Average latency {avg_latency:.2f}ms exceeds 10ms threshold"
    assert p95_latency < 20.0, f"P95 latency {p95_latency:.2f}ms exceeds 20ms threshold"


@pytest.mark.performance
def test_reward_update_latency(strategies):
    """Test that reward updates meet <5ms latency requirement."""
    bandit = BanditController(
        strategies=strategies, alpha_prior=1.0, beta_prior=1.0, warmup_selections=20
    )

    latencies = []
    for _ in range(1000):
        strategy = strategies[_ % len(strategies)]
        start = time.perf_counter()
        bandit.update_reward(strategy, reward=0.75, cost=0.5)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms threshold"
    assert p95_latency < 10.0, f"P95 latency {p95_latency:.2f}ms exceeds 10ms threshold"


@pytest.mark.performance
def test_hyperparameter_sensitivity(evolution_scenario, strategies):
    """Test different hyperparameter configurations."""
    configs = [
        {"alpha_prior": 0.5, "beta_prior": 0.5, "warmup": 10, "name": "aggressive"},
        {"alpha_prior": 1.0, "beta_prior": 1.0, "warmup": 20, "name": "balanced"},
        {"alpha_prior": 2.0, "beta_prior": 2.0, "warmup": 50, "name": "conservative"},
    ]

    results = {}
    for config in configs:
        bandit = BanditController(
            strategies=strategies,
            alpha_prior=config["alpha_prior"],
            beta_prior=config["beta_prior"],
            warmup_selections=config["warmup"],
        )
        scenario = MockEvolutionScenario(seed=42)
        res = run_experiment(bandit, scenario, num_tasks=50)
        results[config["name"]] = res

    balanced = results["balanced"]
    assert balanced["avg_fitness"] >= 0.50, "Balanced config should achieve good fitness"
    assert (
        balanced["cost_efficiency"] >= 0.7
    ), "Balanced config should achieve good cost efficiency"
