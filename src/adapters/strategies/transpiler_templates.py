"""
Code templates for transpiling DSL operations to Python.

Contains optimized Python code patterns for each DSL operation type,
focusing on numpy vectorization and performance.
"""

from typing import Any

# Geometric operation templates
GEOMETRIC_TEMPLATES = {
    "rotate": {
        90: "np.rot90({input}, k=3)",  # numpy rotates counter-clockwise
        180: "np.rot90({input}, k=2)",
        270: "np.rot90({input}, k=1)",
        -90: "np.rot90({input}, k=1)",  # -90 = 270 CCW
    },
    "mirror": {
        "horizontal": "{input}[::-1, :]",
        "vertical": "{input}[:, ::-1]",
        "h": "{input}[::-1, :]",  # alias
        "v": "{input}[:, ::-1]",  # alias
    },
    "flip": {
        "horizontal": "{input}[::-1, :]",
        "vertical": "{input}[:, ::-1]",
        "diagonal": "{input}.T",
        "diagonal_main": "{input}.T",
        "diagonal_anti": "np.flipud(np.fliplr({input}.T))",
        "h": "{input}[::-1, :]",  # alias
        "v": "{input}[:, ::-1]",  # alias
        "d": "{input}.T",  # alias
    },
    "translate": "np.roll({input}, shift=({dy}, {dx}), axis=(0, 1))",
    "crop": "{input}[{y1}:{y2}, {x1}:{x2}]",
    "pad": "np.pad({input}, (({top}, {bottom}), ({left}, {right})), constant_values={value})",
    # Aliases
    "rotateoperation": {
        90: "np.rot90({input}, k=3)",
        180: "np.rot90({input}, k=2)",
        270: "np.rot90({input}, k=1)",
        -90: "np.rot90({input}, k=1)",
    },
    "flipoperation": {
        "horizontal": "{input}[::-1, :]",
        "vertical": "{input}[:, ::-1]",
        "diagonal": "{input}.T",
        "diagonal_main": "{input}.T",
        "diagonal_anti": "np.flipud(np.fliplr({input}.T))",
        "h": "{input}[::-1, :]",
        "v": "{input}[:, ::-1]",
        "d": "{input}.T",
    },
    "translateoperation": "np.roll({input}, shift=({dy}, {dx}), axis=(0, 1))",
    "cropoperation": "{input}[{y1}:{y2}, {x1}:{x2}]",
    "padoperation": "np.pad({input}, (({top}, {bottom}), ({left}, {right})), constant_values={value})",
}

# Color operation templates
COLOR_TEMPLATES = {
    "map": """
    mapping = np.array({mapping_array})
    result = mapping[{input}]
    """,
    "filter": "np.where(np.isin({input}, {colors}), {input}, 0)",
    "mask": "np.where({condition}, {true_val}, {false_val})",
    "replace": "np.where({input} == {old_color}, {new_color}, {input})",
    "invert": "9 - {input}",  # 9-complement for ARC colors
    "threshold": "np.where({input} >= {threshold}, {high_color}, {low_color})",
    # Aliases for operation names
    "colormap": """
    mapping = np.array({mapping_array})
    result = mapping[{input}]
    """,
    "colorfilter": "np.where(np.isin({input}, {colors}), {input}, 0)",
    "colorreplace": "np.where({input} == {old_color}, {new_color}, {input})",
    "colorinvert": "9 - {input}",
    "colorthreshold": "np.where({input} >= {threshold}, {high_color}, {low_color})",
}

# Pattern operation templates
PATTERN_TEMPLATES = {
    "detect": """
    # Pattern detection using sliding window
    pattern = np.array({pattern})
    matches = []
    for y in range({input}.shape[0] - pattern.shape[0] + 1):
        for x in range({input}.shape[1] - pattern.shape[1] + 1):
            if np.array_equal({input}[y:y+pattern.shape[0], x:x+pattern.shape[1]], pattern):
                matches.append((y, x))
    matches
    """,
    "fill": """
    # Flood fill implementation
    def flood_fill(grid, start_y, start_x, new_color):
        grid = grid.copy()
        old_color = grid[start_y, start_x]
        if old_color == new_color:
            return grid

        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == old_color:
                grid[y, x] = new_color
                stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
        return grid

    flood_fill({input}, {start_y}, {start_x}, {color})
    """,
    "match": """
    # Pattern matching - find all occurrences
    pattern = np.array({pattern})
    matches = []
    for y in range({input}.shape[0] - pattern.shape[0] + 1):
        for x in range({input}.shape[1] - pattern.shape[1] + 1):
            if np.array_equal({input}[y:y+pattern.shape[0], x:x+pattern.shape[1]], pattern):
                matches.append((y, x))
    # Store metadata for executor to pick up
    _operation_metadata.update({{'matches': matches, 'match_count': len(matches)}})
    {input}  # Return original grid
    """,
    "replace": """
    # Pattern replacement (avoid overlapping replacements)
    source_pattern = np.array({source_pattern})
    target_pattern = np.array({target_pattern})
    result = {input}.copy()
    replacements = 0
    replaced = set()  # Track positions that have been replaced

    for y in range(result.shape[0] - source_pattern.shape[0] + 1):
        for x in range(result.shape[1] - source_pattern.shape[1] + 1):
            # Skip if this position overlaps with a previous replacement
            if any((y + dy, x + dx) in replaced
                   for dy in range(source_pattern.shape[0])
                   for dx in range(source_pattern.shape[1])):
                continue

            if np.array_equal(result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]], source_pattern):
                result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]] = target_pattern
                # Mark all positions in this pattern as replaced
                for dy in range(source_pattern.shape[0]):
                    for dx in range(source_pattern.shape[1]):
                        replaced.add((y + dy, x + dx))
                replacements += 1

    # Store metadata for executor to pick up
    _operation_metadata.update({{'replacements': replacements}})
    grid = result
    """,
    # Aliases
    "patternfill": """
    # Flood fill implementation
    def flood_fill(grid, start_y, start_x, new_color):
        grid = grid.copy()
        old_color = grid[start_y, start_x]
        if old_color == new_color:
            return grid

        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == old_color:
                grid[y, x] = new_color
                stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
        return grid

    flood_fill({input}, {start_y}, {start_x}, {color})
    """,
    "patternmatch": """
    # Pattern matching - find all occurrences
    pattern = np.array({pattern})
    matches = []
    for y in range({input}.shape[0] - pattern.shape[0] + 1):
        for x in range({input}.shape[1] - pattern.shape[1] + 1):
            if np.array_equal({input}[y:y+pattern.shape[0], x:x+pattern.shape[1]], pattern):
                matches.append((y, x))
    # Store metadata for executor to pick up
    _operation_metadata.update({{'matches': matches, 'match_count': len(matches)}})
    {input}  # Return original grid
    """,
    "pattern_match": """
    # Pattern matching - find all occurrences (with optional mask support)
    pattern = np.array({pattern})
    mask = {mask}
    matches = []

    for y in range({input}.shape[0] - pattern.shape[0] + 1):
        for x in range({input}.shape[1] - pattern.shape[1] + 1):
            window = {input}[y:y+pattern.shape[0], x:x+pattern.shape[1]]
            if mask is not None:
                # Use mask to check only specific positions
                mask_array = np.array(mask)
                # Only check positions where mask is True
                masked_window = window[mask_array]
                masked_pattern = pattern[mask_array]
                pattern_matches = np.array_equal(masked_window, masked_pattern)
            else:
                # Check all positions
                pattern_matches = np.array_equal(window, pattern)

            if pattern_matches:
                matches.append((y, x))
    # Store metadata for executor to pick up
    _operation_metadata.update({{'matches': matches, 'match_count': len(matches)}})
    {input}  # Return original grid
    """,
    "patternreplace": """
    # Pattern replacement (avoid overlapping replacements)
    source_pattern = np.array({source_pattern})
    target_pattern = np.array({target_pattern})
    result = {input}.copy()
    replacements = 0
    replaced = set()  # Track positions that have been replaced

    for y in range(result.shape[0] - source_pattern.shape[0] + 1):
        for x in range(result.shape[1] - source_pattern.shape[1] + 1):
            # Skip if this position overlaps with a previous replacement
            if any((y + dy, x + dx) in replaced
                   for dy in range(source_pattern.shape[0])
                   for dx in range(source_pattern.shape[1])):
                continue

            if np.array_equal(result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]], source_pattern):
                result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]] = target_pattern
                # Mark all positions in this pattern as replaced
                for dy in range(source_pattern.shape[0]):
                    for dx in range(source_pattern.shape[1]):
                        replaced.add((y + dy, x + dx))
                replacements += 1

    # Store metadata for executor to pick up
    _operation_metadata.update({{'replacements': replacements}})
    grid = result
    """,
    "pattern_replace": """
    # Pattern replacement (avoid overlapping replacements)
    source_pattern = np.array({source_pattern})
    target_pattern = np.array({target_pattern})
    result = {input}.copy()
    replacements = 0
    replaced = set()  # Track positions that have been replaced

    for y in range(result.shape[0] - source_pattern.shape[0] + 1):
        for x in range(result.shape[1] - source_pattern.shape[1] + 1):
            # Skip if this position overlaps with a previous replacement
            if any((y + dy, x + dx) in replaced
                   for dy in range(source_pattern.shape[0])
                   for dx in range(source_pattern.shape[1])):
                continue

            if np.array_equal(result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]], source_pattern):
                result[y:y+source_pattern.shape[0], x:x+source_pattern.shape[1]] = target_pattern
                # Mark all positions in this pattern as replaced
                for dy in range(source_pattern.shape[0]):
                    for dx in range(source_pattern.shape[1]):
                        replaced.add((y + dy, x + dx))
                replacements += 1

    # Store metadata for executor to pick up
    _operation_metadata.update({{'replacements': replacements}})
    grid = result
    """,
    "floodfill": """
    # Flood fill implementation
    def flood_fill(grid, start_y, start_x, new_color):
        grid = grid.copy()
        old_color = grid[start_y, start_x]
        if old_color == new_color:
            return grid

        stack = [(start_y, start_x)]
        while stack:
            y, x = stack.pop()
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == old_color:
                grid[y, x] = new_color
                stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
        return grid

    flood_fill({input}, {start_y}, {start_x}, {color})
    """,
}

# Composition operation templates
COMPOSITION_TEMPLATES = {
    "overlay": "np.where({mask} != 0, {overlay}, {base})",
    "extract": "{input}[{y1}:{y2}, {x1}:{x2}].copy()",
    "compose": """
    # Compose multiple grids
    result = {base}.copy()
    for grid in {grids}:
        result = np.where(grid != 0, grid, result)
    result
    """,
}

# Symmetry operation templates
SYMMETRY_TEMPLATES = {
    "check_horizontal": "np.array_equal({input}, {input}[::-1, :])",
    "check_vertical": "np.array_equal({input}, {input}[:, ::-1])",
    "check_diagonal": "np.array_equal({input}, {input}.T)",
    "make_symmetric": """
    # Create symmetric grid
    h, w = {input}.shape
    if '{axis}' == 'horizontal':
        result = np.zeros((h, w), dtype=np.int32)
        result[:h//2] = {input}[:h//2]
        result[h//2:] = result[:h//2][::-1]
    elif '{axis}' == 'vertical':
        result = np.zeros((h, w), dtype=np.int32)
        result[:, :w//2] = {input}[:, :w//2]
        result[:, w//2:] = result[:, :w//2][:, ::-1]
    result
    """,
    "create_symmetry": """
    # Create symmetric grid by mirroring
    h, w = {input}.shape
    if '{axis}' == 'horizontal':
        # Mirror horizontally
        result = np.vstack([{input}, {input}[::-1, :]])
    elif '{axis}' == 'vertical':
        # Mirror vertically
        result = np.hstack([{input}, {input}[:, ::-1]])
    elif '{axis}' == 'diagonal':
        # Create diagonal symmetry
        result = {input} + {input}.T
    else:
        result = {input}
    result
    """,
    # Aliases
    "createsymmetry": """
    # Create symmetric grid by mirroring
    h, w = {input}.shape
    if '{axis}' == 'horizontal':
        # Mirror horizontally
        result = np.vstack([{input}, {input}[::-1, :]])
    elif '{axis}' == 'vertical':
        # Mirror vertically
        result = np.hstack([{input}, {input}[:, ::-1]])
    elif '{axis}' == 'diagonal':
        # Create diagonal symmetry
        result = {input} + {input}.T
    else:
        result = {input}
    result
    """,
}

# Connectivity operation templates
CONNECTIVITY_TEMPLATES = {
    "components": """
    # Find connected components
    from scipy import ndimage
    labeled, num_features = ndimage.label({input} != {background})
    components = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        components.append(mask.astype(int) * {input}[mask][0])
    components
    """,
    "largest_component": """
    # Find largest connected component
    from scipy import ndimage
    labeled, num_features = ndimage.label({input} != {background})
    if num_features == 0:
        result = np.zeros_like({input})
    else:
        sizes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])
        largest_idx = np.argmax(sizes) + 1
        result = np.where(labeled == largest_idx, {input}, 0)
    result
    """,
    "filter_components": """
    # Filter connected components by size
    from scipy import ndimage
    labeled, num_features = ndimage.label({input} != {background})
    result = np.zeros_like({input})
    for i in range(1, num_features + 1):
        mask = labeled == i
        size = np.sum(mask)
        if {min_size} <= size <= {max_size}:
            result[mask] = {input}[mask][0]
    result
    """,
    # Aliases
    "connectedcomponents": """
    # Find connected components
    from scipy import ndimage
    labeled, num_features = ndimage.label({input} != {background})
    components = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        components.append(mask.astype(int) * {input}[mask][0])
    components
    """,
    "filtercomponents": """
    # Filter connected components by size
    from scipy import ndimage
    labeled, num_features = ndimage.label({input} != {background})
    result = np.zeros_like({input})
    for i in range(1, num_features + 1):
        mask = labeled == i
        size = np.sum(mask)
        if {min_size} <= size <= {max_size}:
            result[mask] = {input}[mask][0]
    result
    """,
}

# Edge detection templates
EDGE_TEMPLATES = {
    "detect_edges": """
    # Simple edge detection
    h, w = {input}.shape
    edges = np.zeros_like({input})
    for y in range(h):
        for x in range(w):
            if {input}[y, x] != 0:
                # Check neighbors
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and {input}[ny, nx] == 0:
                        edges[y, x] = {input}[y, x]
                        break
    edges
    """,
    "boundary": """
    # Get boundary pixels
    from scipy import ndimage
    struct = np.ones((3, 3))
    eroded = ndimage.binary_erosion({input} != 0, struct)
    boundary = ({input} != 0) & ~eroded
    np.where(boundary, {input}, 0)
    """,
    "boundary_tracing": """
    # Trace outer boundary of objects
    h, w = {input}.shape
    boundary = np.zeros_like({input})
    # Find boundary pixels - those with at least one empty neighbor
    for y in range(h):
        for x in range(w):
            if {input}[y, x] != 0:
                has_empty_neighbor = False
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if {input}[ny, nx] == 0:
                            has_empty_neighbor = True
                            break
                    else:
                        # Edge of grid counts as empty
                        has_empty_neighbor = True
                        break
                if has_empty_neighbor:
                    boundary[y, x] = {input}[y, x]
    boundary
    """,
    "contour_extraction": """
    # Extract contours (connected boundaries)
    from scipy import ndimage
    # Get all boundary pixels
    struct = np.ones((3, 3))
    eroded = ndimage.binary_erosion({input} != 0, struct)
    boundary = ({input} != 0) & ~eroded
    # Label connected boundary components
    labeled, num_features = ndimage.label(boundary)
    contours = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        contour = np.where(mask, {input}[mask][0], 0)
        contours.append(contour)
    contours
    """,
    # Aliases
    "edgedetection": """
    # Simple edge detection
    h, w = {input}.shape
    edges = np.zeros_like({input})
    for y in range(h):
        for x in range(w):
            if {input}[y, x] != 0:
                # Check neighbors
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and {input}[ny, nx] == 0:
                        edges[y, x] = {input}[y, x]
                        break
    edges
    """,
    "boundarytracing": """
    # Trace outer boundary of objects
    h, w = {input}.shape
    boundary = np.zeros_like({input})
    # Find boundary pixels - those with at least one empty neighbor
    for y in range(h):
        for x in range(w):
            if {input}[y, x] != 0:
                has_empty_neighbor = False
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if {input}[ny, nx] == 0:
                            has_empty_neighbor = True
                            break
                    else:
                        # Edge of grid counts as empty
                        has_empty_neighbor = True
                        break
                if has_empty_neighbor:
                    boundary[y, x] = {input}[y, x]
    boundary
    """,
    "contourextraction": """
    # Extract contours (connected boundaries)
    from scipy import ndimage
    # Get all boundary pixels
    struct = np.ones((3, 3))
    eroded = ndimage.binary_erosion({input} != 0, struct)
    boundary = ({input} != 0) & ~eroded
    # Label connected boundary components
    labeled, num_features = ndimage.label(boundary)
    contours = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        contour = np.where(mask, {input}[mask][0], 0)
        contours.append(contour)
    contours
    """,
}


def get_all_templates() -> dict[str, dict[str, Any]]:
    """Get all operation templates organized by category."""
    return {
        "geometric": GEOMETRIC_TEMPLATES,
        "color": COLOR_TEMPLATES,
        "pattern": PATTERN_TEMPLATES,
        "composition": COMPOSITION_TEMPLATES,
        "symmetry": SYMMETRY_TEMPLATES,
        "connectivity": CONNECTIVITY_TEMPLATES,
        "edges": EDGE_TEMPLATES,
    }
