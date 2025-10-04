"""Enhanced Program Synthesis Strategy with Smart Model Routing integration."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.adapters.external.gemini_client import GeminiClient
from src.adapters.external.glm_client import GLMClient
from src.adapters.external.gpt5_client import GPT5Client
from src.adapters.external.llm_cache_manager import LLMCacheManager
from src.adapters.external.local_model_client import LocalModelClient
from src.adapters.external.qwen_client import QwenClient
from src.adapters.external.smart_model_router import SmartModelRouter
from src.adapters.strategies.program_synthesis import (
    ProgramGenerationStrategy,
    ProgramSynthesisAdapter,
    ProgramSynthesisConfig,
)
from src.domain.dsl.base import DSLProgram
from src.domain.models import ARCTask, ARCTaskSolution, StrategyType
from src.domain.services.dsl_engine import DSLEngine
from src.infrastructure.components.budget_controller import create_default_budget_controller
from src.infrastructure.config import Config

logger = logging.getLogger(__name__)


@dataclass
class EnhancedProgramSynthesisConfig(ProgramSynthesisConfig):
    """Extended configuration with LLM routing support."""

    # LLM-guided generation settings
    use_llm_generation: bool = True
    llm_generation_ratio: float = 0.3  # Proportion of LLM-generated programs
    llm_batch_size: int = 5  # Number of programs to generate per LLM call

    # Smart routing settings
    budget_limit: float = 100.0
    cache_similarity_threshold: float = 0.85
    enable_caching: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("data/llm_program_cache"))

    # Model selection overrides
    force_model_tier: str | None = None  # Force specific model tier
    disable_fallback: bool = False  # Disable local model fallback

    # Performance settings
    parallel_llm_calls: int = 3  # Max concurrent LLM calls
    llm_timeout: float = 30.0  # Timeout per LLM call


class EnhancedProgramSynthesisAdapter(ProgramSynthesisAdapter):
    """Program synthesis with intelligent LLM routing for program generation."""

    def __init__(
        self,
        config: EnhancedProgramSynthesisConfig | None = None,
        engine: DSLEngine | None = None
    ):
        # Use enhanced config
        config = config or EnhancedProgramSynthesisConfig()
        super().__init__(config, engine)

        self.enhanced_config = config

        # Initialize LLM components if enabled
        if config.use_llm_generation:
            self._initialize_llm_components()
        else:
            self.model_router = None
            self.cache_manager = None

    def _initialize_llm_components(self):
        """Initialize LLM routing and caching components."""
        # Create budget controller
        self.budget_controller = create_default_budget_controller(
            self.enhanced_config.budget_limit
        )

        # Create cache manager
        self.cache_manager = LLMCacheManager(
            cache_dir=self.enhanced_config.cache_dir,
            similarity_threshold=self.enhanced_config.cache_similarity_threshold
        )

        # Create model router
        app_config = Config()  # Use app config for platform detection
        self.model_router = SmartModelRouter(
            config=app_config,
            budget_controller=self.budget_controller,
            cache_dir=self.enhanced_config.cache_dir
        )

        # Register LLM providers
        self._register_llm_providers()

    def _register_llm_providers(self):
        """Register all LLM providers with the router."""
        providers = [
            ("qwen2.5-coder-32b", QwenClient(budget_controller=self.budget_controller)),
            ("gemini-2.5-flash", GeminiClient(budget_controller=self.budget_controller)),
            ("glm-4.5", GLMClient(budget_controller=self.budget_controller)),
            ("gpt-5", GPT5Client(budget_controller=self.budget_controller)),
            ("falcon-mamba-7b-local", LocalModelClient(budget_controller=self.budget_controller))
        ]

        for model_id, client in providers:
            try:
                self.model_router.register_provider(model_id, client)
                logger.info(f"Registered LLM provider: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to register {model_id}: {e}")

    def solve(self, task: ARCTask) -> ARCTaskSolution:
        """Solve task with LLM-enhanced program synthesis."""
        # Check if we need to run async LLM generation
        if self.enhanced_config.use_llm_generation and self.model_router:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._solve_async(task))
            finally:
                loop.close()
        else:
            # Fall back to base implementation
            return super().solve(task)

    async def _solve_async(self, task: ARCTask) -> ARCTaskSolution:
        """Async implementation of solve with LLM integration."""
        start_time = datetime.now()
        synthesis_start = time.time()

        logger.info(f"Starting enhanced program synthesis for task: {task.task_id}")

        try:
            # Analyze task
            analysis = self._analyze_task(task)

            # Generate programs with LLM assistance
            candidate_programs = await self._generate_enhanced_programs(task, analysis)

            # Evaluate programs
            best_program, evaluation_results = self._evaluate_programs(
                candidate_programs, task
            )

            if best_program is None:
                logger.warning(f"No successful program found for task: {task.task_id}")
                return self._create_failure_solution(task, start_time)

            # Execute on test input
            final_result = self._execute_program_on_test(best_program, task)

            synthesis_time = time.time() - synthesis_start

            # Enhanced metadata
            metadata = {
                "program_synthesis": True,
                "generation_strategy": "llm_enhanced",
                "best_program": self.serializer.serialize_program(best_program),
                "programs_evaluated": len(candidate_programs),
                "synthesis_time": synthesis_time,
                "task_analysis": analysis,
                "evaluation_results": evaluation_results,
                "llm_usage": await self._get_llm_usage_stats(),
                **self.generation_stats
            }

            confidence = self._calculate_confidence(evaluation_results)

            return ARCTaskSolution(
                task_id=task.task_id,
                predictions=[final_result.grid] if final_result.success else [task.test_input],
                strategy_used=StrategyType.PROGRAM_SYNTHESIS,
                confidence_score=confidence,
                metadata=metadata,
                resource_usage=self._calculate_resource_usage(task, start_time, synthesis_time)
            )

        except Exception as e:
            logger.error(f"Enhanced program synthesis failed for task {task.task_id}: {e}")
            return self._create_failure_solution(task, start_time, error=str(e))

    async def _generate_enhanced_programs(
        self,
        task: ARCTask,
        analysis: dict[str, Any]
    ) -> list[DSLProgram]:
        """Generate programs with LLM assistance based on complexity."""
        programs = []

        # Always include template-based programs as baseline
        template_programs = self._generate_template_based_programs(task, analysis)
        programs.extend(template_programs)

        # Determine how many LLM programs to generate
        total_attempts = self.config.max_generation_attempts
        llm_attempts = int(total_attempts * self.enhanced_config.llm_generation_ratio)

        if llm_attempts > 0:
            # Generate LLM programs in batches
            llm_programs = await self._generate_llm_programs(
                task, analysis, llm_attempts
            )
            programs.extend(llm_programs)

        # Add search-based programs if configured
        if self.config.generation_strategy in [
            ProgramGenerationStrategy.SEARCH_BASED,
            ProgramGenerationStrategy.HYBRID
        ]:
            search_programs = self._generate_search_based_programs(task, analysis)
            programs.extend(search_programs)

        return programs[:self.config.max_generation_attempts]

    async def _generate_llm_programs(
        self,
        task: ARCTask,
        analysis: dict[str, Any],
        num_programs: int
    ) -> list[DSLProgram]:
        """Generate programs using LLM with smart routing."""
        programs = []

        # Prepare task description for LLM
        task_desc = self._format_task_for_llm(task, analysis)

        # Generate in batches
        batch_size = self.enhanced_config.llm_batch_size
        num_batches = (num_programs + batch_size - 1) // batch_size

        # Process batches concurrently
        tasks = []
        for i in range(num_batches):
            batch_num = min(batch_size, num_programs - i * batch_size)
            prompt = self._create_program_generation_prompt(task_desc, batch_num)

            # Create async task for batch
            task_future = self._generate_program_batch(task, prompt, i)
            tasks.append(task_future)

            # Limit concurrent calls
            if len(tasks) >= self.enhanced_config.parallel_llm_calls:
                batch_results = await asyncio.gather(*tasks)
                for result in batch_results:
                    programs.extend(result)
                tasks = []

        # Process remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            for result in batch_results:
                programs.extend(result)

        return programs

    async def _generate_program_batch(
        self,
        task: ARCTask,
        prompt: str,
        batch_id: int
    ) -> list[DSLProgram]:
        """Generate a batch of programs using LLM."""
        try:
            # Get response with routing
            response, routing_decision, metadata = await self.model_router.generate_with_routing(
                task=task,
                prompt=prompt,
                override_tier=self.enhanced_config.force_model_tier,
                use_cache=self.enhanced_config.enable_caching
            )

            # Log routing decision
            logger.info(
                f"Batch {batch_id} routed to {routing_decision.model_tier.name} "
                f"(complexity: {routing_decision.complexity_score:.2f})"
            )

            # Parse programs from response
            programs = self._parse_llm_programs(response)

            return programs

        except Exception as e:
            logger.error(f"Error generating program batch {batch_id}: {e}")
            return []

    def _format_task_for_llm(self, task: ARCTask, analysis: dict[str, Any]) -> str:
        """Format task information for LLM prompt."""
        lines = ["ARC Task Analysis:"]

        # Add grid dimensions
        dims = task.get_grid_dimensions()
        lines.append(f"- Input dimensions: {dims['train_input']}")
        lines.append(f"- Output dimensions: {dims['train_output']}")

        # Add analysis insights
        if analysis.get("color_changes"):
            lines.append(f"- Color changes detected: {analysis['color_changes']}")
        if analysis.get("spatial_changes"):
            lines.append(f"- Spatial transformations: {analysis['spatial_changes']}")

        # Add training examples
        lines.append("\nTraining examples:")
        for i, example in enumerate(task.train_examples[:3]):  # Limit to 3 examples
            lines.append(f"\nExample {i+1}:")
            lines.append(f"Input grid shape: {len(example['input'])}x{len(example['input'][0])}")
            lines.append(f"Output grid shape: {len(example['output'])}x{len(example['output'][0])}")

            # Add color distribution
            input_colors = {cell for row in example['input'] for cell in row}
            output_colors = {cell for row in example['output'] for cell in row}
            lines.append(f"Input colors: {sorted(input_colors)}")
            lines.append(f"Output colors: {sorted(output_colors)}")

        return "\n".join(lines)

    def _create_program_generation_prompt(self, task_desc: str, num_programs: int) -> str:
        """Create prompt for LLM program generation."""
        return f"""Generate {num_programs} DSL program(s) to solve this ARC task.

{task_desc}

Available DSL operations:
- Geometric: rotate(angle), flip(direction), translate(dx,dy), crop(x,y,w,h), pad(size,value)
- Color: color_map(mapping), color_filter(color), color_replace(from,to), color_invert()
- Pattern: pattern_fill(pattern), pattern_match(template), pattern_replace(from,to), flood_fill(x,y,color)
- Composition: overlay(mode), extract_region(x,y,w,h), concatenate(grids,axis), resize(w,h,method)

Generate Python-style DSL programs that transform input to output. Each program should be a sequence of operations.

Example format:
```
program1 = [
    rotate(90),
    color_replace(1, 2),
    flip("horizontal")
]
```

Generate {num_programs} different approach(es):"""

    def _parse_llm_programs(self, response: str) -> list[DSLProgram]:
        """Parse DSL programs from LLM response."""
        programs = []

        try:
            # Extract code blocks
            import re
            code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)

            for block in code_blocks:
                # Parse operations from block
                operations = self._parse_operations_from_code(block)
                if operations:
                    program = DSLProgram(operations=operations)
                    programs.append(program)

            # Also try to parse inline operations
            if not programs:
                operations = self._parse_operations_from_code(response)
                if operations:
                    programs.append(DSLProgram(operations=operations))

        except Exception as e:
            logger.warning(f"Error parsing LLM programs: {e}")

        return programs

    def _parse_operations_from_code(self, code: str) -> list[Any]:
        """Parse DSL operations from code text."""
        operations = []

        # Simple regex-based parsing for common operations
        operation_patterns = {
            r'rotate\((\d+)\)': lambda m: self._create_rotate_op(int(m.group(1))),
            r'flip\("?(horizontal|vertical)"?\)': lambda m: self._create_flip_op(m.group(1)),
            r'color_replace\((\d+),\s*(\d+)\)': lambda m: self._create_color_replace_op(
                int(m.group(1)), int(m.group(2))
            ),
            r'color_filter\((\d+)\)': lambda m: self._create_color_filter_op(int(m.group(1))),
            r'translate\((-?\d+),\s*(-?\d+)\)': lambda m: self._create_translate_op(
                int(m.group(1)), int(m.group(2))
            ),
            r'flood_fill\((\d+),\s*(\d+),\s*(\d+)\)': lambda m: self._create_flood_fill_op(
                int(m.group(1)), int(m.group(2)), int(m.group(3))
            ),
        }

        import re
        for pattern, creator in operation_patterns.items():
            for match in re.finditer(pattern, code):
                try:
                    op = creator(match)
                    if op:
                        operations.append(op)
                except Exception as e:
                    logger.debug(f"Failed to create operation from {match.group()}: {e}")

        return operations

    # Operation creation helpers
    def _create_rotate_op(self, angle: int):
        from src.domain.dsl.geometric import RotateOperation
        return RotateOperation(angle=angle)

    def _create_flip_op(self, direction: str):
        from src.domain.dsl.geometric import FlipOperation
        axis = 0 if direction == "horizontal" else 1
        return FlipOperation(axis=axis)

    def _create_color_replace_op(self, from_color: int, to_color: int):
        from src.domain.dsl.color import ColorReplaceOperation
        return ColorReplaceOperation(from_color=from_color, to_color=to_color)

    def _create_color_filter_op(self, color: int):
        from src.domain.dsl.color import ColorFilterOperation
        return ColorFilterOperation(target_color=color)

    def _create_translate_op(self, dx: int, dy: int):
        from src.domain.dsl.geometric import TranslateOperation
        return TranslateOperation(dx=dx, dy=dy)

    def _create_flood_fill_op(self, x: int, y: int, color: int):
        from src.domain.dsl.pattern import FloodFillOperation
        return FloodFillOperation(start_position=(x, y), new_color=color)

    async def _get_llm_usage_stats(self) -> dict[str, Any]:
        """Get LLM usage statistics."""
        if not self.model_router:
            return {}

        budget_summary = self.budget_controller.get_usage_summary()
        performance_summary = self.model_router.get_performance_summary()
        cache_stats = self.cache_manager.get_statistics() if self.cache_manager else {}

        return {
            "budget_used": budget_summary["total_cost"],
            "budget_remaining": budget_summary["remaining_budget"],
            "model_performance": performance_summary["model_performance"],
            "cache_statistics": cache_stats
        }


def create_enhanced_synthesis_adapter(
    budget_limit: float = 100.0,
    use_llm: bool = True
) -> EnhancedProgramSynthesisAdapter:
    """Factory function to create enhanced synthesis adapter."""
    config = EnhancedProgramSynthesisConfig(
        use_llm_generation=use_llm,
        budget_limit=budget_limit,
        generation_strategy=ProgramGenerationStrategy.HYBRID,
        max_generation_attempts=100,  # More attempts with LLM assistance
        llm_generation_ratio=0.3,  # 30% LLM-generated
        cache_similarity_threshold=0.85
    )

    return EnhancedProgramSynthesisAdapter(config)
