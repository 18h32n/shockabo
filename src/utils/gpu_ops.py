"""
GPU-accelerated vectorized operations for DSL transformations.

This module provides PyTorch-based vectorized implementations of common DSL operations
that can be applied to batches of grids efficiently on GPU.
"""


import structlog
import torch
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class VectorizedOps:
    """
    Vectorized operations for batch processing of grid transformations on GPU.

    All operations work on batch tensors of shape (batch_size, height, width).
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        """
        Initialize vectorized operations.

        Args:
            device: Device to run operations on
        """
        self.device = device

    def rotate_90(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Rotate batch of grids 90 degrees clockwise.

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Rotated grids of shape (batch_size, width, height)
        """
        # Rotate 90 clockwise: transpose and flip vertically
        # Original: (B, H, W) -> Transpose: (B, W, H) -> Flip along last dim: (B, W, H)
        return torch.flip(grids.transpose(1, 2), dims=[2])

    def rotate_180(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Rotate batch of grids 180 degrees.

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Rotated grids of shape (batch_size, height, width)
        """
        # Rotate 180: flip both dimensions
        return torch.flip(grids, dims=[1, 2])

    def rotate_270(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Rotate batch of grids 270 degrees clockwise (90 counterclockwise).

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Rotated grids of shape (batch_size, width, height)
        """
        # Rotate 270 clockwise: flip horizontally then transpose
        return torch.flip(grids, dims=[2]).transpose(1, 2)

    def flip_horizontal(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Flip batch of grids horizontally (left-right mirror).

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Flipped grids of shape (batch_size, height, width)
        """
        return torch.flip(grids, dims=[2])

    def flip_vertical(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Flip batch of grids vertically (up-down mirror).

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Flipped grids of shape (batch_size, height, width)
        """
        return torch.flip(grids, dims=[1])

    def flip_main_diagonal(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Flip batch of grids along main diagonal (transpose).

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Flipped grids of shape (batch_size, width, height)
        """
        return grids.transpose(1, 2)

    def flip_anti_diagonal(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Flip batch of grids along anti-diagonal.

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Flipped grids of shape (batch_size, width, height)
        """
        # Flip along anti-diagonal: rotate 180 then transpose
        return self.rotate_180(grids).transpose(1, 2)

    def translate(
        self,
        grids: torch.Tensor,
        shift_y: int,
        shift_x: int,
        fill_value: float = 0
    ) -> torch.Tensor:
        """
        Translate batch of grids by specified offset.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            shift_y: Vertical shift (positive = down)
            shift_x: Horizontal shift (positive = right)
            fill_value: Value to fill empty spaces

        Returns:
            Translated grids of shape (batch_size, height, width)
        """
        batch_size, height, width = grids.shape

        # Use roll for periodic boundary, then mask out wrapped regions
        rolled = torch.roll(grids, shifts=(shift_y, shift_x), dims=(1, 2))

        # Create mask for valid regions (not wrapped)
        mask = torch.ones_like(rolled, dtype=torch.bool)

        if shift_y > 0:
            mask[:, :shift_y, :] = False
        elif shift_y < 0:
            mask[:, shift_y:, :] = False

        if shift_x > 0:
            mask[:, :, :shift_x] = False
        elif shift_x < 0:
            mask[:, :, shift_x:] = False

        # Apply mask and fill
        result = torch.where(mask, rolled, fill_value)
        return result

    def map_colors(
        self,
        grids: torch.Tensor,
        color_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Map colors in batch of grids using a color mapping.

        Args:
            grids: Tensor of shape (batch_size, height, width) with integer values
            color_map: Tensor of shape (num_colors,) mapping old to new colors

        Returns:
            Grids with mapped colors
        """
        # Ensure grids are long type for indexing
        grids_long = grids.long()

        # Apply color mapping using indexing
        # Clamp to valid range
        max_color = color_map.shape[0] - 1
        clamped = torch.clamp(grids_long, 0, max_color)

        # Map colors
        return color_map[clamped]

    def filter_by_color(
        self,
        grids: torch.Tensor,
        target_color: int,
        replacement: float = 0
    ) -> torch.Tensor:
        """
        Filter grids to keep only specific color.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            target_color: Color to keep
            replacement: Value to replace non-target colors

        Returns:
            Filtered grids
        """
        mask = grids == target_color
        return torch.where(mask, grids, replacement)

    def extract_region(
        self,
        grids: torch.Tensor,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
        pad_value: float = 0
    ) -> torch.Tensor:
        """
        Extract rectangular region from batch of grids.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            y_start, y_end: Vertical bounds (inclusive start, exclusive end)
            x_start, x_end: Horizontal bounds (inclusive start, exclusive end)
            pad_value: Value for padding if region extends outside grid

        Returns:
            Extracted regions
        """
        batch_size, height, width = grids.shape

        # Handle negative indices
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        y_end = min(height, y_end)
        x_end = min(width, x_end)

        # Extract region
        if y_start < y_end and x_start < x_end:
            return grids[:, y_start:y_end, x_start:x_end]
        else:
            # Return empty region
            return torch.full((batch_size, 0, 0), pad_value, device=grids.device)

    def pad_grids(
        self,
        grids: torch.Tensor,
        target_height: int,
        target_width: int,
        pad_value: float = 0,
        align: str = "top_left"
    ) -> torch.Tensor:
        """
        Pad batch of grids to target size.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            target_height: Target height after padding
            target_width: Target width after padding
            pad_value: Value to use for padding
            align: Alignment ("top_left", "center", etc.)

        Returns:
            Padded grids of shape (batch_size, target_height, target_width)
        """
        batch_size, height, width = grids.shape

        if height >= target_height and width >= target_width:
            return grids[:, :target_height, :target_width]

        # Calculate padding amounts based on alignment
        if align == "center":
            pad_top = (target_height - height) // 2
            pad_bottom = target_height - height - pad_top
            pad_left = (target_width - width) // 2
            pad_right = target_width - width - pad_left
        else:  # top_left
            pad_top = 0
            pad_bottom = max(0, target_height - height)
            pad_left = 0
            pad_right = max(0, target_width - width)

        # Apply padding (left, right, top, bottom)
        return F.pad(grids, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    def count_colors(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Count occurrences of each color in batch of grids.

        Args:
            grids: Tensor of shape (batch_size, height, width)

        Returns:
            Tensor of shape (batch_size, num_colors) with color counts
        """
        batch_size = grids.shape[0]
        max_color = int(grids.max().item()) + 1

        # Flatten spatial dimensions
        flat = grids.view(batch_size, -1).long()

        # Count colors using one-hot encoding
        counts = torch.zeros(batch_size, max_color, device=grids.device)

        for b in range(batch_size):
            counts[b] = torch.bincount(flat[b], minlength=max_color).float()

        return counts

    def get_bounding_box(
        self,
        grids: torch.Tensor,
        target_color: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get bounding box for non-background pixels or specific color.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            target_color: If specified, get bounding box of this color only

        Returns:
            Tuple of (y_min, y_max, x_min, x_max) tensors, each of shape (batch_size,)
        """
        batch_size, height, width = grids.shape

        # Create mask for pixels of interest
        if target_color is not None:
            mask = grids == target_color
        else:
            mask = grids != 0  # Assuming 0 is background

        # Get indices where mask is True
        y_mins = torch.full((batch_size,), height, device=grids.device)
        y_maxs = torch.full((batch_size,), -1, device=grids.device)
        x_mins = torch.full((batch_size,), width, device=grids.device)
        x_maxs = torch.full((batch_size,), -1, device=grids.device)

        for b in range(batch_size):
            if mask[b].any():
                rows, cols = torch.where(mask[b])
                y_mins[b] = rows.min()
                y_maxs[b] = rows.max()
                x_mins[b] = cols.min()
                x_maxs[b] = cols.max()

        return y_mins, y_maxs, x_mins, x_maxs

    def apply_mask(
        self,
        grids: torch.Tensor,
        mask: torch.Tensor,
        fill_value: float = 0
    ) -> torch.Tensor:
        """
        Apply binary mask to batch of grids.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            mask: Binary mask of same shape
            fill_value: Value for masked-out pixels

        Returns:
            Masked grids
        """
        return torch.where(mask.bool(), grids, fill_value)

    def compose_grids(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        mask: torch.Tensor | None = None,
        mode: str = "overlay"
    ) -> torch.Tensor:
        """
        Compose two batches of grids.

        Args:
            background: Background grids
            foreground: Foreground grids
            mask: Optional mask for selective composition
            mode: Composition mode ("overlay", "add", "multiply")

        Returns:
            Composed grids
        """
        if mode == "overlay":
            if mask is not None:
                return torch.where(mask.bool(), foreground, background)
            else:
                # Use non-zero foreground pixels
                mask = foreground != 0
                return torch.where(mask, foreground, background)
        elif mode == "add":
            return torch.clamp(background + foreground, 0, 9)
        elif mode == "multiply":
            return background * foreground
        else:
            raise ValueError(f"Unknown composition mode: {mode}")

    def detect_patterns(
        self,
        grids: torch.Tensor,
        pattern: torch.Tensor,
        threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Detect occurrences of pattern in batch of grids using convolution.

        Args:
            grids: Tensor of shape (batch_size, height, width)
            pattern: Pattern tensor of shape (pattern_height, pattern_width)
            threshold: Matching threshold (1.0 = exact match)

        Returns:
            Detection map of shape (batch_size, height, width)
        """
        grids.shape[0]
        ph, pw = pattern.shape

        # Convert to float for convolution
        grids_float = grids.float()
        pattern_float = pattern.float()

        # Reshape for conv2d (need channel dimension)
        grids_4d = grids_float.unsqueeze(1)  # (B, 1, H, W)
        kernel = pattern_float.unsqueeze(0).unsqueeze(0)  # (1, 1, pH, pW)

        # Convolve to find matches
        conv_result = F.conv2d(grids_4d, kernel, padding=0)

        # Normalize by pattern sum to get match score
        pattern_sum = pattern_float.sum()
        if pattern_sum > 0:
            match_scores = conv_result / pattern_sum
        else:
            match_scores = conv_result

        # Threshold to get binary detection map
        detections = (match_scores >= threshold).float()

        # Remove channel dimension
        return detections.squeeze(1)
