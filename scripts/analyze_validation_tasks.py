"""Analyze selected validation tasks for patterns and metadata."""

import json
from collections import Counter
from pathlib import Path


def analyze_colors(task: dict) -> set[int]:
    """Get unique colors used in task."""
    colors = set()
    for example in task.get("train", []):
        for row in example["input"]:
            colors.update(row)
        for row in example["output"]:
            colors.update(row)
    return colors

def detect_transformation_pattern(task: dict) -> list[str]:
    """Heuristically detect transformation patterns."""
    patterns = []

    for example in task.get("train", []):
        inp = example["input"]
        out = example["output"]

        in_h, in_w = len(inp), len(inp[0]) if inp else 0
        out_h, out_w = len(out), len(out[0]) if out else 0

        if in_h == out_h and in_w == out_w:
            patterns.append("same_size")
        elif out_h > in_h or out_w > in_w:
            patterns.append("expansion")
        elif out_h < in_h or out_w < in_w:
            patterns.append("reduction")

        in_colors = set()
        out_colors = set()
        for row in inp:
            in_colors.update(row)
        for row in out:
            out_colors.update(row)

        if in_colors != out_colors:
            if len(out_colors) < len(in_colors):
                patterns.append("color_reduction")
            else:
                patterns.append("color_addition")

        if out_h == in_h * 2 and out_w == in_w * 2:
            patterns.append("doubling")
        elif out_h == in_h // 2 and out_w == in_w // 2:
            patterns.append("halving")

    return list(set(patterns))

def main():
    """Analyze validation tasks."""
    project_root = Path(__file__).parent.parent
    input_file = project_root / "tests" / "performance" / "data" / "synthesis_validation_tasks.json"

    with open(input_file, encoding="utf-8", errors="replace") as f:
        tasks = json.load(f)

    print(f"Analyzing {len(tasks)} tasks...\n")

    size_dist = Counter()
    pattern_dist = Counter()
    color_dist = Counter()

    for task_data in tasks:
        task = task_data["task"]
        task_id = task_data["id"]
        category = task_data["category"]

        size_dist[category] += 1

        colors = analyze_colors(task)
        color_dist[len(colors)] += 1

        patterns = detect_transformation_pattern(task)
        for p in patterns:
            pattern_dist[p] += 1

        print(f"{task_id} ({category}):")
        print(f"  Size: {task_data['max_height']}x{task_data['max_width']}")
        print(f"  Colors: {len(colors)} unique")
        print(f"  Training examples: {len(task.get('train', []))}")
        print(f"  Test examples: {len(task.get('test', []))}")
        print(f"  Patterns: {', '.join(patterns) if patterns else 'unknown'}")
        print()

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("\nSize Distribution:")
    for size, count in sorted(size_dist.items()):
        print(f"  {size}: {count} tasks ({count/len(tasks)*100:.0f}%)")

    print("\nColor Complexity:")
    for num_colors, count in sorted(color_dist.items()):
        print(f"  {num_colors} colors: {count} tasks")

    print("\nTransformation Patterns:")
    for pattern, count in pattern_dist.most_common():
        print(f"  {pattern}: {count} tasks")

    print("\nTask Selection Criteria:")
    print("  - Grid sizes: 3x3 to 29x29 (covers full range)")
    print(f"  - Size distribution: {dict(size_dist)}")
    print(f"  - Color complexity: {min(color_dist.keys())}-{max(color_dist.keys())} unique colors")
    print(f"  - Total training examples: {sum(len(t['task'].get('train', [])) for t in tasks)}")
    print(f"  - Total test examples: {sum(len(t['task'].get('test', [])) for t in tasks)}")

    updated_tasks = []
    for task_data in tasks:
        task = task_data["task"]
        colors = analyze_colors(task)
        patterns = detect_transformation_pattern(task)

        task_data["metadata"] = {
            "num_colors": len(colors),
            "num_train_examples": len(task.get("train", [])),
            "num_test_examples": len(task.get("test", [])),
            "detected_patterns": patterns,
            "difficulty": "hard" if task_data["category"] == "xlarge" else
                         "medium" if task_data["category"] in ["medium", "large"] else "easy"
        }
        updated_tasks.append(task_data)

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(updated_tasks, f, indent=2)

    print(f"\nUpdated {input_file} with metadata")

if __name__ == "__main__":
    main()
