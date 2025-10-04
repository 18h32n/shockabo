"""Select 20 diverse ARC tasks for synthesis validation testing."""

import json
import random
from pathlib import Path
from typing import Any

random.seed(42)

def load_task(task_path: Path) -> dict[str, Any]:
    """Load a single ARC task."""
    with open(task_path, encoding="utf-8", errors="replace") as f:
        return json.load(f)

def get_grid_size(grid: list[list[int]]) -> tuple[int, int]:
    """Get (height, width) of a grid."""
    return (len(grid), len(grid[0]) if grid else 0)

def get_max_grid_size(task: dict[str, Any]) -> tuple[int, int]:
    """Get maximum grid size across all examples in a task."""
    max_height = 0
    max_width = 0

    for example in task.get("train", []):
        h, w = get_grid_size(example["input"])
        max_height = max(max_height, h)
        max_width = max(max_width, w)
        h, w = get_grid_size(example["output"])
        max_height = max(max_height, h)
        max_width = max(max_width, w)

    for example in task.get("test", []):
        h, w = get_grid_size(example["input"])
        max_height = max(max_height, h)
        max_width = max(max_width, w)

    return (max_height, max_width)

def classify_size_category(height: int, width: int) -> str:
    """Classify task by grid size."""
    max_dim = max(height, width)
    if max_dim <= 10:
        return "small"
    elif max_dim <= 20:
        return "medium"
    elif max_dim <= 25:
        return "large"
    else:
        return "xlarge"

def select_diverse_tasks(training_dir: Path, target_count: int = 20) -> list[dict[str, Any]]:
    """Select diverse tasks across size categories."""

    task_files = list(training_dir.glob("*.json"))
    print(f"Found {len(task_files)} training tasks")

    tasks_by_category = {
        "small": [],
        "medium": [],
        "large": [],
        "xlarge": []
    }

    for task_file in task_files:
        try:
            task = load_task(task_file)
            h, w = get_max_grid_size(task)
            category = classify_size_category(h, w)

            tasks_by_category[category].append({
                "id": task_file.stem,
                "file": str(task_file),
                "height": h,
                "width": w,
                "category": category,
                "task": task
            })
        except Exception as e:
            print(f"Error loading {task_file}: {e}")

    for cat, tasks in tasks_by_category.items():
        print(f"{cat}: {len(tasks)} tasks")

    target_distribution = {
        "small": 4,
        "medium": 8,
        "large": 5,
        "xlarge": 3
    }

    selected = []
    for category, count in target_distribution.items():
        available = tasks_by_category[category]
        if len(available) >= count:
            selected.extend(random.sample(available, count))
        else:
            selected.extend(available)
            print(f"Warning: Only {len(available)} tasks in {category} category (wanted {count})")

    return selected

def main():
    """Select and save validation tasks."""
    project_root = Path(__file__).parent.parent
    training_dir = project_root / "data" / "arc-agi" / "training"
    output_dir = project_root / "tests" / "performance" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_tasks = select_diverse_tasks(training_dir, target_count=20)

    print(f"\nSelected {len(selected_tasks)} tasks:")
    for task_info in sorted(selected_tasks, key=lambda x: (x["category"], x["id"])):
        print(f"  {task_info['id']} ({task_info['category']}): {task_info['height']}x{task_info['width']}")

    output_data = []
    for task_info in selected_tasks:
        output_data.append({
            "id": task_info["id"],
            "category": task_info["category"],
            "max_height": task_info["height"],
            "max_width": task_info["width"],
            "task": task_info["task"]
        })

    output_file = output_dir / "synthesis_validation_tasks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
