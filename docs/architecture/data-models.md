# 4. Data Models

## Core Domain Models

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class StrategyType(str, Enum):
    TEST_TIME_TRAINING = "ttt"
    PROGRAM_SYNTHESIS = "program_synthesis"
    EVOLUTION = "evolution"
    IMITATION_LEARNING = "imitation"

@dataclass
class ARCTask:
    """Core ARC task representation"""
    task_id: str
    task_source: str  # 'training', 'evaluation', 'test'
    difficulty_level: str  # 'easy', 'medium', 'hard', 'unknown'
    train_examples: List[Dict[str, List[List[int]]]]
    test_input: List[List[int]]
    test_output: Optional[List[List[int]]] = None
    family_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None

@dataclass
class TaskSubmission:
    """User submission for a task"""
    submission_id: str
    task_id: str
    user_id: str
    predicted_output: List[List[int]]
    strategy_used: StrategyType
    confidence_score: float
    processing_time_ms: int
    resource_usage: Dict[str, float]
    metadata: Dict[str, Any]
    submitted_at: datetime

@dataclass
class ExperimentRun:
    """Track experiment execution"""
    run_id: str
    experiment_name: str
    task_ids: List[str]
    strategy_config: Dict[str, Any]
    metrics: Dict[str, float]
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_log: Optional[str] = None

@dataclass
class LLMCache:
    """Cache LLM responses for efficiency"""
    cache_id: str
    prompt_hash: str
    model_name: str
    temperature: float
    response_text: str
    token_count: int
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None

@dataclass 
class ResourceUsage:
    """Track resource consumption"""
    task_id: str
    strategy_type: StrategyType
    cpu_seconds: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    api_calls: Dict[str, int]
    total_tokens: int
    estimated_cost: float
    timestamp: datetime

@dataclass
class TTTAdaptation:
    """Store Test-Time Training adaptations"""
    adaptation_id: str
    task_id: str
    base_model_checkpoint: str
    adapted_weights_path: str
    training_examples: List[Dict[str, Any]]
    adaptation_metrics: Dict[str, float]
    created_at: datetime

@dataclass
class ExperimentTracker:
    """Track experiment metadata and results"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    strategy_configs: Dict[str, Any]
    task_selection_criteria: Dict[str, Any]
    results: Dict[str, Any]
    insights: List[str]
    created_at: datetime
    updated_at: datetime
```
