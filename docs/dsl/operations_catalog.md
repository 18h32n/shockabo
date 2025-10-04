# ARC DSL Operations Catalog

This document catalogs 50+ core grid operations for the ARC Domain-Specific Language (DSL), organized by category and prioritized by frequency and importance in puzzle-solving contexts.

## Table of Contents

1. [Geometric Transformations](#geometric-transformations) (12 operations)
2. [Color Operations](#color-operations) (10 operations)
3. [Pattern Detection & Manipulation](#pattern-detection--manipulation) (12 operations)
4. [Grid Composition & Decomposition](#grid-composition--decomposition) (8 operations)
5. [Connectivity & Topology](#connectivity--topology) (8 operations)
6. [Symmetry Operations](#symmetry-operations) (6 operations)
7. [Spatial Analysis](#spatial-analysis) (8 operations)
8. [Size & Scaling](#size--scaling) (4 operations)

**Total Operations: 68**

---

## Geometric Transformations
*Priority: CRITICAL - These are fundamental transformations appearing in 70%+ of ARC tasks*

### T001: Rotate 90° Clockwise
- **Description**: Rotates the entire grid 90 degrees clockwise
- **Input**: Grid[H,W] → Grid[W,H]
- **Example**: `rotate_90_cw(grid)`
- **Frequency**: Very High
- **Use Cases**: Orientation changes, pattern alignment

### T002: Rotate 180°
- **Description**: Rotates the entire grid 180 degrees
- **Input**: Grid[H,W] → Grid[H,W]
- **Example**: `rotate_180(grid)`
- **Frequency**: High
- **Use Cases**: Inversion puzzles, symmetry creation

### T003: Rotate 270° Clockwise (90° CCW)
- **Description**: Rotates the entire grid 270 degrees clockwise
- **Input**: Grid[H,W] → Grid[W,H]
- **Example**: `rotate_270_cw(grid)`
- **Frequency**: High
- **Use Cases**: Completing rotation sequences

### T004: Mirror Horizontal
- **Description**: Reflects grid horizontally (left-right flip)
- **Input**: Grid[H,W] → Grid[H,W]
- **Example**: `mirror_horizontal(grid)`
- **Frequency**: Very High
- **Use Cases**: Symmetry puzzles, reflection patterns

### T005: Mirror Vertical
- **Description**: Reflects grid vertically (top-bottom flip)
- **Input**: Grid[H,W] → Grid[H,W]
- **Example**: `mirror_vertical(grid)`
- **Frequency**: Very High
- **Use Cases**: Symmetry puzzles, reflection patterns

### T006: Mirror Diagonal (Main)
- **Description**: Reflects along main diagonal (transpose)
- **Input**: Grid[H,W] → Grid[W,H]
- **Example**: `mirror_diagonal_main(grid)`
- **Frequency**: Medium
- **Use Cases**: Matrix transpose operations

### T007: Mirror Diagonal (Anti)
- **Description**: Reflects along anti-diagonal
- **Input**: Grid[H,W] → Grid[W,H]
- **Example**: `mirror_diagonal_anti(grid)`
- **Frequency**: Medium
- **Use Cases**: Complex symmetry patterns

### T008: Translate
- **Description**: Shifts grid content by offset (dx, dy)
- **Input**: Grid, dx: int, dy: int → Grid
- **Example**: `translate(grid, dx=2, dy=-1)`
- **Frequency**: High
- **Use Cases**: Pattern alignment, moving objects

### T009: Crop
- **Description**: Extracts rectangular subregion
- **Input**: Grid, x: int, y: int, width: int, height: int → Grid
- **Example**: `crop(grid, x=1, y=1, width=5, height=3)`
- **Frequency**: High
- **Use Cases**: Focus on specific regions, object extraction

### T010: Pad
- **Description**: Adds border around grid with specified color
- **Input**: Grid, pad_size: int, fill_color: int → Grid
- **Example**: `pad(grid, pad_size=2, fill_color=0)`
- **Frequency**: Medium
- **Use Cases**: Creating borders, expanding workspace

### T011: Scale Up
- **Description**: Enlarges grid by integer factor
- **Input**: Grid, factor: int → Grid
- **Example**: `scale_up(grid, factor=3)`
- **Frequency**: Medium
- **Use Cases**: Magnification puzzles, detail enhancement

### T012: Scale Down
- **Description**: Reduces grid by integer factor
- **Input**: Grid, factor: int → Grid
- **Example**: `scale_down(grid, factor=2)`
- **Frequency**: Medium
- **Use Cases**: Compression puzzles, overview creation

---

## Color Operations
*Priority: CRITICAL - Color manipulation is central to most ARC tasks*

### C001: Replace Color
- **Description**: Replaces all instances of one color with another
- **Input**: Grid, old_color: int, new_color: int → Grid
- **Example**: `replace_color(grid, old_color=1, new_color=5)`
- **Frequency**: Very High
- **Use Cases**: Color substitution, pattern modification

### C002: Map Colors
- **Description**: Applies color mapping dictionary
- **Input**: Grid, color_map: Dict[int, int] → Grid
- **Example**: `map_colors(grid, {1: 3, 2: 7, 3: 1})`
- **Frequency**: High
- **Use Cases**: Complex color transformations

### C003: Invert Colors
- **Description**: Applies color inversion (0→9, 1→8, etc.)
- **Input**: Grid → Grid
- **Example**: `invert_colors(grid)`
- **Frequency**: Medium
- **Use Cases**: Negative image effects

### C004: Filter by Color
- **Description**: Keeps only specified colors, others become background
- **Input**: Grid, colors: Set[int], bg_color: int → Grid
- **Example**: `filter_by_color(grid, colors={1, 3, 5}, bg_color=0)`
- **Frequency**: High
- **Use Cases**: Isolating specific patterns

### C005: Mask by Color
- **Description**: Creates binary mask where specified color is 1
- **Input**: Grid, target_color: int → Grid[0|1]
- **Example**: `mask_by_color(grid, target_color=3)`
- **Frequency**: High
- **Use Cases**: Shape extraction, region identification

### C006: Threshold Colors
- **Description**: Maps colors below/above threshold to binary values
- **Input**: Grid, threshold: int, low_val: int, high_val: int → Grid
- **Example**: `threshold_colors(grid, threshold=5, low_val=0, high_val=9)`
- **Frequency**: Medium
- **Use Cases**: Brightness-based operations

### C007: Gradient Fill
- **Description**: Fills region with color gradient
- **Input**: Grid, start_pos: Tuple[int, int], end_pos: Tuple[int, int], start_color: int, end_color: int → Grid
- **Example**: `gradient_fill(grid, start_pos=(0,0), end_pos=(5,5), start_color=1, end_color=9)`
- **Frequency**: Low
- **Use Cases**: Smooth transitions

### C008: Noise Add
- **Description**: Adds random color noise to grid
- **Input**: Grid, probability: float, noise_colors: Set[int] → Grid
- **Example**: `add_noise(grid, probability=0.1, noise_colors={1,2,3})`
- **Frequency**: Low
- **Use Cases**: Pattern corruption, testing robustness

### C009: Dominant Color
- **Description**: Returns most frequent non-background color
- **Input**: Grid, bg_color: int → int
- **Example**: `dominant_color(grid, bg_color=0)`
- **Frequency**: Medium
- **Use Cases**: Pattern analysis, color statistics

### C010: Unique Colors
- **Description**: Returns set of all colors present in grid
- **Input**: Grid → Set[int]
- **Example**: `unique_colors(grid)`
- **Frequency**: Medium
- **Use Cases**: Color palette analysis

---

## Pattern Detection & Manipulation
*Priority: HIGH - Pattern recognition and manipulation is key to ARC solving*

### P001: Flood Fill
- **Description**: Fills connected region with specified color
- **Input**: Grid, start_pos: Tuple[int, int], new_color: int → Grid
- **Example**: `flood_fill(grid, start_pos=(2,3), new_color=7)`
- **Frequency**: Very High
- **Use Cases**: Region coloring, connected component marking

### P002: Fill Shape
- **Description**: Fills interior of closed shape with color
- **Input**: Grid, boundary_color: int, fill_color: int → Grid
- **Example**: `fill_shape(grid, boundary_color=1, fill_color=5)`
- **Frequency**: High
- **Use Cases**: Shape completion, interior filling

### P003: Extract Objects
- **Description**: Identifies and extracts distinct objects
- **Input**: Grid, bg_color: int → List[Grid]
- **Example**: `extract_objects(grid, bg_color=0)`
- **Frequency**: Very High
- **Use Cases**: Object isolation, multi-object processing

### P004: Find Pattern
- **Description**: Locates all instances of pattern in grid
- **Input**: Grid, pattern: Grid → List[Tuple[int, int]]
- **Example**: `find_pattern(grid, pattern=small_template)`
- **Frequency**: High
- **Use Cases**: Template matching, pattern detection

### P005: Replace Pattern
- **Description**: Replaces all pattern instances with replacement
- **Input**: Grid, pattern: Grid, replacement: Grid → Grid
- **Example**: `replace_pattern(grid, pattern=old_shape, replacement=new_shape)`
- **Frequency**: High
- **Use Cases**: Shape substitution, pattern transformation

### P006: Tile Pattern
- **Description**: Repeats pattern across grid
- **Input**: pattern: Grid, target_width: int, target_height: int → Grid
- **Example**: `tile_pattern(pattern, target_width=15, target_height=10)`
- **Frequency**: Medium
- **Use Cases**: Wallpaper patterns, tessellation

### P007: Complete Pattern
- **Description**: Completes partial pattern based on detected structure
- **Input**: Grid → Grid
- **Example**: `complete_pattern(partial_grid)`
- **Frequency**: Medium
- **Use Cases**: Pattern completion puzzles

### P008: Remove Noise
- **Description**: Removes isolated pixels or small artifacts
- **Input**: Grid, min_size: int, bg_color: int → Grid
- **Example**: `remove_noise(grid, min_size=3, bg_color=0)`
- **Frequency**: Medium
- **Use Cases**: Cleaning artifacts, noise removal

### P009: Outline Shape
- **Description**: Creates outline of filled shapes
- **Input**: Grid, outline_color: int, bg_color: int → Grid
- **Example**: `outline_shape(grid, outline_color=1, bg_color=0)`
- **Frequency**: Medium
- **Use Cases**: Edge detection, shape outlining

### P010: Solid Fill
- **Description**: Fills all empty spaces with specified color
- **Input**: Grid, target_color: int, fill_color: int → Grid
- **Example**: `solid_fill(grid, target_color=0, fill_color=5)`
- **Frequency**: High
- **Use Cases**: Background filling

### P011: Skeletonize
- **Description**: Reduces shapes to their skeletal structure
- **Input**: Grid, shape_color: int, bg_color: int → Grid
- **Example**: `skeletonize(grid, shape_color=1, bg_color=0)`
- **Frequency**: Low
- **Use Cases**: Shape simplification

### P012: Pattern Frequency
- **Description**: Counts occurrences of pattern in grid
- **Input**: Grid, pattern: Grid → int
- **Example**: `pattern_frequency(grid, pattern=search_template)`
- **Frequency**: Medium
- **Use Cases**: Pattern counting, frequency analysis

---

## Grid Composition & Decomposition
*Priority: HIGH - Essential for multi-grid operations*

### G001: Overlay
- **Description**: Combines two grids with transparency rules
- **Input**: Grid, overlay: Grid, transparent_color: int → Grid
- **Example**: `overlay(base_grid, overlay_grid, transparent_color=0)`
- **Frequency**: High
- **Use Cases**: Layer combination, additive operations

### G002: Mask Apply
- **Description**: Applies binary mask to grid
- **Input**: Grid, mask: Grid[0|1], mask_color: int → Grid
- **Example**: `apply_mask(grid, mask, mask_color=0)`
- **Frequency**: High
- **Use Cases**: Selective operations, region masking

### G003: Concatenate Horizontal
- **Description**: Joins grids side by side
- **Input**: Grid, Grid → Grid
- **Example**: `concat_horizontal(left_grid, right_grid)`
- **Frequency**: Medium
- **Use Cases**: Grid assembly, horizontal joining

### G004: Concatenate Vertical
- **Description**: Joins grids top to bottom
- **Input**: Grid, Grid → Grid
- **Example**: `concat_vertical(top_grid, bottom_grid)`
- **Frequency**: Medium
- **Use Cases**: Grid assembly, vertical joining

### G005: Split Horizontal
- **Description**: Divides grid into left and right parts
- **Input**: Grid, split_pos: int → Tuple[Grid, Grid]
- **Example**: `split_horizontal(grid, split_pos=5)`
- **Frequency**: Medium
- **Use Cases**: Grid decomposition

### G006: Split Vertical
- **Description**: Divides grid into top and bottom parts
- **Input**: Grid, split_pos: int → Tuple[Grid, Grid]
- **Example**: `split_vertical(grid, split_pos=3)`
- **Frequency**: Medium
- **Use Cases**: Grid decomposition

### G007: Difference
- **Description**: Computes difference between two grids
- **Input**: Grid, Grid, diff_color: int → Grid
- **Example**: `grid_difference(grid1, grid2, diff_color=9)`
- **Frequency**: Medium
- **Use Cases**: Change detection, diff visualization

### G008: Intersection
- **Description**: Finds common elements between grids
- **Input**: Grid, Grid, bg_color: int → Grid
- **Example**: `grid_intersection(grid1, grid2, bg_color=0)`
- **Frequency**: Medium
- **Use Cases**: Common pattern extraction

---

## Connectivity & Topology
*Priority: HIGH - Essential for understanding spatial relationships*

### N001: Connected Components
- **Description**: Labels connected regions with unique IDs
- **Input**: Grid, bg_color: int, connectivity: int → Grid
- **Example**: `connected_components(grid, bg_color=0, connectivity=4)`
- **Frequency**: Very High
- **Use Cases**: Object separation, region labeling

### N002: Count Components
- **Description**: Returns number of connected components
- **Input**: Grid, bg_color: int, connectivity: int → int
- **Example**: `count_components(grid, bg_color=0, connectivity=8)`
- **Frequency**: High
- **Use Cases**: Object counting

### N003: Largest Component
- **Description**: Extracts largest connected component
- **Input**: Grid, bg_color: int → Grid
- **Example**: `largest_component(grid, bg_color=0)`
- **Frequency**: High
- **Use Cases**: Main object extraction

### N004: Component Size
- **Description**: Returns size of each connected component
- **Input**: Grid, bg_color: int → Dict[int, int]
- **Example**: `component_sizes(grid, bg_color=0)`
- **Frequency**: Medium
- **Use Cases**: Size-based filtering

### N005: Boundary Detection
- **Description**: Finds boundary pixels of objects
- **Input**: Grid, obj_color: int, boundary_color: int → Grid
- **Example**: `detect_boundary(grid, obj_color=1, boundary_color=9)`
- **Frequency**: High
- **Use Cases**: Edge finding, contour detection

### N006: Interior Detection
- **Description**: Finds interior pixels of objects
- **Input**: Grid, obj_color: int → Grid
- **Example**: `detect_interior(grid, obj_color=1)`
- **Frequency**: Medium
- **Use Cases**: Interior processing

### N007: Shortest Path
- **Description**: Finds shortest path between two points
- **Input**: Grid, start: Tuple[int, int], end: Tuple[int, int], obstacle_colors: Set[int] → List[Tuple[int, int]]
- **Example**: `shortest_path(grid, start=(0,0), end=(5,5), obstacle_colors={1,2})`
- **Frequency**: Medium
- **Use Cases**: Pathfinding, route planning

### N008: Neighbors
- **Description**: Gets neighboring cells of a position
- **Input**: Grid, pos: Tuple[int, int], connectivity: int → List[Tuple[int, int]]
- **Example**: `get_neighbors(grid, pos=(3,4), connectivity=8)`
- **Frequency**: High
- **Use Cases**: Local analysis, neighbor checking

---

## Symmetry Operations
*Priority: MEDIUM - Important for symmetry-based puzzles*

### S001: Detect Horizontal Symmetry
- **Description**: Checks if grid has horizontal line symmetry
- **Input**: Grid → bool
- **Example**: `has_horizontal_symmetry(grid)`
- **Frequency**: Medium
- **Use Cases**: Symmetry validation

### S002: Detect Vertical Symmetry
- **Description**: Checks if grid has vertical line symmetry
- **Input**: Grid → bool
- **Example**: `has_vertical_symmetry(grid)`
- **Frequency**: Medium
- **Use Cases**: Symmetry validation

### S003: Detect Rotational Symmetry
- **Description**: Checks if grid has rotational symmetry
- **Input**: Grid, angle: int → bool
- **Example**: `has_rotational_symmetry(grid, angle=90)`
- **Frequency**: Medium
- **Use Cases**: Rotational pattern detection

### S004: Create Horizontal Symmetry
- **Description**: Makes grid horizontally symmetric by copying half
- **Input**: Grid, copy_direction: str → Grid
- **Example**: `create_horizontal_symmetry(grid, copy_direction='left_to_right')`
- **Frequency**: Medium
- **Use Cases**: Symmetry completion

### S005: Create Vertical Symmetry
- **Description**: Makes grid vertically symmetric by copying half
- **Input**: Grid, copy_direction: str → Grid
- **Example**: `create_vertical_symmetry(grid, copy_direction='top_to_bottom')`
- **Frequency**: Medium
- **Use Cases**: Symmetry completion

### S006: Symmetry Axis
- **Description**: Finds axis of symmetry in grid
- **Input**: Grid → Optional[Tuple[str, int]]
- **Example**: `find_symmetry_axis(grid)`
- **Frequency**: Low
- **Use Cases**: Symmetry analysis

---

## Spatial Analysis
*Priority: MEDIUM - Useful for understanding spatial relationships*

### A001: Centroid
- **Description**: Calculates center of mass for colored regions
- **Input**: Grid, target_color: int → Tuple[float, float]
- **Example**: `calculate_centroid(grid, target_color=1)`
- **Frequency**: Medium
- **Use Cases**: Center finding, balance point

### A002: Bounding Box
- **Description**: Finds minimal rectangle containing all non-background pixels
- **Input**: Grid, bg_color: int → Tuple[int, int, int, int]
- **Example**: `bounding_box(grid, bg_color=0)`
- **Frequency**: High
- **Use Cases**: Object bounds, cropping guides

### A003: Distance Transform
- **Description**: Computes distance to nearest target pixel
- **Input**: Grid, target_color: int → Grid[float]
- **Example**: `distance_transform(grid, target_color=1)`
- **Frequency**: Low
- **Use Cases**: Distance analysis

### A004: Convex Hull
- **Description**: Finds convex hull of colored pixels
- **Input**: Grid, target_color: int → List[Tuple[int, int]]
- **Example**: `convex_hull(grid, target_color=1)`
- **Frequency**: Low
- **Use Cases**: Shape analysis

### A005: Object Alignment
- **Description**: Aligns objects to grid or each other
- **Input**: Grid, alignment: str → Grid
- **Example**: `align_objects(grid, alignment='center')`
- **Frequency**: Medium
- **Use Cases**: Layout organization

### A006: Grid Statistics
- **Description**: Computes statistics about grid content
- **Input**: Grid → Dict[str, Any]
- **Example**: `grid_statistics(grid)`
- **Frequency**: Medium
- **Use Cases**: Grid analysis, debugging

### A007: Density Map
- **Description**: Creates density heatmap of colored pixels
- **Input**: Grid, target_color: int, kernel_size: int → Grid[float]
- **Example**: `density_map(grid, target_color=1, kernel_size=3)`
- **Frequency**: Low
- **Use Cases**: Density analysis

### A008: Aspect Ratio
- **Description**: Calculates width/height ratio of objects
- **Input**: Grid, target_color: int → float
- **Example**: `aspect_ratio(grid, target_color=1)`
- **Frequency**: Medium
- **Use Cases**: Shape analysis

---

## Size & Scaling
*Priority: MEDIUM - Useful for size-based operations*

### Z001: Resize
- **Description**: Changes grid dimensions to specific size
- **Input**: Grid, new_width: int, new_height: int, method: str → Grid
- **Example**: `resize(grid, new_width=10, new_height=8, method='nearest')`
- **Frequency**: Medium
- **Use Cases**: Size normalization

### Z002: Fit to Size
- **Description**: Scales grid to fit within maximum dimensions
- **Input**: Grid, max_width: int, max_height: int → Grid
- **Example**: `fit_to_size(grid, max_width=15, max_height=15)`
- **Frequency**: Medium
- **Use Cases**: Size constraints

### Z003: Trim Empty
- **Description**: Removes empty border around content
- **Input**: Grid, bg_color: int → Grid
- **Example**: `trim_empty(grid, bg_color=0)`
- **Frequency**: High
- **Use Cases**: Content focusing, size reduction

### Z004: Normalize Size
- **Description**: Standardizes all objects to same size
- **Input**: Grid, target_size: int → Grid
- **Example**: `normalize_size(grid, target_size=3)`
- **Frequency**: Low
- **Use Cases**: Size standardization

---

## Operation Priorities Summary

### CRITICAL Priority (Must implement first - 22 operations):
- **Geometric**: T001-T012 (12 ops) - Basic transformations
- **Color**: C001-C005 (5 ops) - Essential color operations
- **Pattern**: P001, P003, P010 (3 ops) - Core pattern operations  
- **Connectivity**: N001, N005 (2 ops) - Basic connectivity

### HIGH Priority (Second phase - 24 operations):
- **Color**: C006-C010 (5 ops) - Advanced color operations
- **Pattern**: P002, P004-P009, P012 (8 ops) - Pattern manipulation
- **Composition**: G001-G008 (8 ops) - Grid combination operations
- **Connectivity**: N002-N004, N008 (3 ops) - Advanced connectivity

### MEDIUM Priority (Third phase - 22 operations):
- **Pattern**: P007, P011 (2 ops) - Complex pattern operations
- **Connectivity**: N006-N007 (2 ops) - Advanced topology
- **Symmetry**: S001-S006 (6 ops) - Symmetry operations
- **Spatial**: A001-A002, A005-A006, A008 (5 ops) - Basic spatial analysis
- **Size**: Z001-Z004 (4 ops) - Size operations
- **Color**: C007-C008 (2 ops) - Specialized color operations
- **Spatial**: A003-A004, A007 (3 ops) - Advanced spatial analysis

This catalog provides a comprehensive foundation for the ARC DSL, covering the essential operations needed for grid-based puzzle solving while maintaining clear categorization and priority ordering for implementation.