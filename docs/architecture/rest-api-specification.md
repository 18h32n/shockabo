# 7. REST API Specification

## API Overview
Base URL: `http://localhost:8000/api/v1`

## Authentication
```http
Authorization: Bearer <token>
```

## Endpoints

### Task Management

**GET /tasks/{task_id}**
```json
Response 200:
{
  "task_id": "arc_2023_001",
  "task_source": "training",
  "difficulty_level": "medium",
  "train_examples": [...],
  "test_input": [[0,1,2], [3,4,5]],
  "family_id": "pattern_completion",
  "metadata": {
    "grid_size": [3, 3],
    "colors_used": 6
  }
}
```

**POST /tasks/submit**
```json
Request:
{
  "task_id": "arc_2023_001",
  "predicted_output": [[1,2,3], [4,5,6]],
  "strategy_preference": "ttt"
}

Response 201:
{
  "submission_id": "sub_12345",
  "task_id": "arc_2023_001",
  "is_correct": true,
  "confidence_score": 0.92,
  "strategy_used": "ttt",
  "processing_time_ms": 2847,
  "resource_usage": {
    "cpu_seconds": 2.3,
    "memory_mb": 512,
    "estimated_cost": 0.03
  }
}
```

### Strategy Management

**POST /strategies/evaluate**
```json
Request:
{
  "task_id": "arc_2023_001",
  "strategies": ["ttt", "program_synthesis"]
}

Response 200:
{
  "evaluations": [
    {
      "strategy": "ttt",
      "estimated_cost": 0.25,
      "estimated_time_ms": 3000,
      "confidence_estimate": 0.85
    },
    {
      "strategy": "program_synthesis", 
      "estimated_cost": 0.10,
      "estimated_time_ms": 1500,
      "confidence_estimate": 0.70
    }
  ],
  "recommendation": "ttt"
}
```

### Experiment Management

**POST /experiments/create**
```json
Request:
{
  "name": "TTT Hyperparameter Search",
  "hypothesis": "Smaller learning rates improve TTT accuracy",
  "task_ids": ["arc_001", "arc_002", "arc_003"],
  "strategy_config": {
    "type": "ttt",
    "learning_rate": 0.0001,
    "num_epochs": 10
  }
}

Response 201:
{
  "experiment_id": "exp_67890",
  "status": "created",
  "estimated_duration_minutes": 45,
  "estimated_cost": 1.26
}
```

**GET /experiments/{experiment_id}/status**
```json
Response 200:
{
  "experiment_id": "exp_67890",
  "status": "in_progress",
  "progress": {
    "completed_tasks": 2,
    "total_tasks": 3,
    "current_task": "arc_003"
  },
  "partial_results": {
    "accuracy": 0.67,
    "avg_confidence": 0.88
  }
}
```

## WebSocket Events

**Connection**
```javascript
const socket = io('ws://localhost:8000', {
  auth: { token: 'Bearer <token>' }
});
```

**Events**
```javascript
// Task processing updates
socket.on('task:progress', (data) => {
  // data: { task_id, stage, progress_percent, message }
});

// Experiment updates
socket.on('experiment:update', (data) => {
  // data: { experiment_id, task_completed, current_metrics }
});

// System alerts
socket.on('system:alert', (data) => {
  // data: { level, message, timestamp }
});
```
