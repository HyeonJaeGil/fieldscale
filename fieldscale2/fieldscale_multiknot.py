from collections import deque
import numpy as np
import cv2
from typing import Iterable, Union, Literal


def get_neighbor_grids(grid: np.ndarray, xy: tuple[int, int], max_distance: int = 1) -> list[tuple[int, int]]:
    h, w = grid.shape
    x, y = xy
    neighbors = [
        (x + i, y + j)
        for i in range(-max_distance, max_distance + 1)
        for j in range(-max_distance, max_distance + 1)
        if (i != 0 or j != 0) and (0 <= x + i < h) and (0 <= y + j < w)
    ]
    return sorted(neighbors, key=lambda item: (item[0], item[1]))



def local_extrema_suppression(
    grid: np.ndarray,
    local_distance: int,
    diff_threshold: float,
    extrema: Literal["max", "min"],
) -> np.ndarray:
    if local_distance <= 0 or diff_threshold <= 0:
        return grid

    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=local_distance)
        neighbor_values = np.array([grid[x, y] for x, y in neighbors])
        if extrema == "max":
            diff = grid[i, j] - neighbor_values.mean()
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() + diff_threshold
        elif extrema == "min":
            diff = neighbor_values.mean() - grid[i, j]
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() - diff_threshold
        else:
            raise ValueError("extrema must be 'max' or 'min'.")
    return grid


def local_extrema_suppression_repeat(
    grid: np.ndarray,
    local_distance: int,
    diff_threshold: float,
    extrema: Literal["max", "min"],
    repeats: int = 5,
) -> np.ndarray:
    output = grid.copy()
    for _ in range(repeats):
        output = local_extrema_suppression(
            output,
            local_distance=local_distance,
            diff_threshold=diff_threshold,
            extrema=extrema,
        )
    return output


def message_passing(grid: np.ndarray, direction: Literal["increase", "decrease"]) -> np.ndarray:
    if direction not in {"increase", "decrease"}:
        raise ValueError("direction must be 'increase' or 'decrease'.")

    grid_new = np.zeros_like(grid, dtype=np.float64)
    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=1)
        neighbor_values = [grid[x, y] for x, y in neighbors]
        mean_value = np.mean(neighbor_values + [grid[i, j]])
        bigger, smaller = (mean_value, grid[i, j]) if mean_value > grid[i, j] else (grid[i, j], mean_value)
        grid_new[i, j] = bigger if direction == "increase" else smaller
    return grid_new


class Fieldscale:
    def __init__(
        self,
        max_diff: float = 400,
        min_diff: float = 400,
        iteration: int = 7,
        gamma: float = 1.5,
        clahe: bool = True,
        video: bool = False,
        ema_period: int = 10,
        ema_alpha_override: float | None = None,
        les_mode: Literal["single", "repeat5"] = "single",
        grid_shape: tuple[int, int] = (8, 8),
    ):
        self.max_diff = float(max_diff)
        self.min_diff = float(min_diff)
        self.iteration = int(iteration)
        self.gamma = float(gamma)
        self.clahe = bool(clahe)
        self.video = bool(video)
        self.ema_period = int(ema_period)
        self.ema_alpha_override = None if ema_alpha_override is None else float(ema_alpha_override)
        self.les_mode = str(les_mode)
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self.ema_alpha = (
            self.ema_alpha_override
            if self.ema_alpha_override is not None
            else 2.0 / (self.ema_period + 1.0)
        )
        self.prev_min_field = None
        self.prev_max_field = None
        self.min_field_buffer = deque(maxlen=self.ema_period)
        self.max_field_buffer = deque(maxlen=self.ema_period)

    def reset_temporal_state(self) -> None:
        self.prev_min_field = None
        self.prev_max_field = None
        self.min_field_buffer.clear()
        self.max_field_buffer.clear()

    def _load_image(self, input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(input, str):
            image = cv2.imread(input, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Unable to read image from path: {input}")
            return image
        if isinstance(input, np.ndarray):
            return input
        raise TypeError("Input should be a file path or a numpy.ndarray.")

    def _apply_les(
        self,
        grid: np.ndarray,
        diff_threshold: float,
        extrema: Literal["max", "min"],
    ) -> np.ndarray:
        if self.les_mode == "repeat5":
            return local_extrema_suppression_repeat(
                grid,
                local_distance=2,
                diff_threshold=diff_threshold,
                extrema=extrema,
                repeats=5,
            )
        return local_extrema_suppression(
            grid,
            local_distance=2,
            diff_threshold=diff_threshold,
            extrema=extrema,
        )

    def _postprocess_output(self, rescaled: np.ndarray) -> np.ndarray:
        output = rescaled
        if self.gamma > 0:
            output = (255.0 * np.power(output.astype(np.float64) / 255.0, self.gamma)).astype(np.uint8)
        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.grid_shape)
            output = clahe.apply(output)
        return output


def parse_percentiles(values: Iterable[float]) -> tuple[float, ...]:
    percentiles = tuple(float(value) for value in values)
    if len(percentiles) < 2:
        raise ValueError("At least two percentiles are required.")
    if any(value < 0.0 or value > 100.0 for value in percentiles):
        raise ValueError("Percentiles must be within [0, 100].")
    if any(b <= a for a, b in zip(percentiles[:-1], percentiles[1:])):
        raise ValueError("Percentiles must be strictly increasing.")
    return percentiles


def gridwise_percentiles(
    image: np.ndarray,
    percentiles: tuple[float, ...],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output = np.zeros((len(percentiles), grid_shape[0], grid_shape[1]), dtype=np.float64)

    for i, j in np.ndindex(grid_shape):
        patch = image[
            patch_shape[0] * i: patch_shape[0] * (i + 1),
            patch_shape[1] * j: patch_shape[1] * (j + 1),
        ].astype(np.float64)
        output[:, i, j] = np.percentile(patch.reshape(-1), percentiles)

    return output


def compute_frame_grid_span_median(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> float:
    lower_grid, upper_grid = gridwise_percentiles(
        image,
        percentiles=(float(lower_percentile), float(upper_percentile)),
        grid_shape=grid_shape,
    ).astype(np.float64)
    return float(np.median(upper_grid - lower_grid))


def select_frame_adaptive_les_diff(frame_grid_span_median: float) -> float:
    frame_grid_span_median = float(frame_grid_span_median)
    if frame_grid_span_median < 100.0:
        return 100.0
    if frame_grid_span_median < 1000.0:
        return 500.0
    if frame_grid_span_median > 2000.0:
        return 800.0
    return 600.0


def symmetric_message_passing(grid: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.float64)
    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=1)
        vals = [grid[x, y] for x, y in neighbors] + [grid[i, j]]
        mu = float(np.mean(vals))
        out[i, j] = grid[i, j] + alpha * (mu - grid[i, j])
    return out


def enforce_knot_order(
    knot_stack: np.ndarray,
    min_interval: float = 1.0,
    mode: Literal["bottom_up", "top_down"] = "bottom_up",
) -> np.ndarray:
    output = knot_stack.astype(np.float64, copy=True)

    for i, j in np.ndindex(output.shape[1:]):
        arr = output[:, i, j].copy()
        if mode == "bottom_up":
            arr = np.maximum.accumulate(arr)
            for k in range(1, len(arr)):
                arr[k] = max(arr[k], arr[k - 1] + min_interval)
        elif mode == "top_down":
            arr = np.minimum.accumulate(arr[::-1])[::-1]
            for k in range(len(arr) - 2, -1, -1):
                arr[k] = min(arr[k], arr[k + 1] - min_interval)
        else:
            raise ValueError(f"Unsupported enforce mode: {mode}")
        output[:, i, j] = arr

    return output


def project_widths_with_caps(
    initial_widths: np.ndarray,
    max_widths: np.ndarray,
    total_width: float = 255.0,
    eps: float = 1e-6,
) -> np.ndarray:
    widths = np.minimum(initial_widths.astype(np.float64), max_widths.astype(np.float64))
    remaining = float(total_width - widths.sum())
    active = (max_widths - widths) > eps

    while remaining > eps and np.any(active):
        share = np.where(active, initial_widths, 0.0)
        share_sum = float(share.sum())
        if share_sum <= eps:
            share = np.where(active, 1.0, 0.0)
            share_sum = float(share.sum())

        proposed = remaining * share / share_sum
        capacity = np.maximum(max_widths - widths, 0.0)
        addition = np.minimum(proposed, capacity)
        delta = float(addition.sum())
        if delta <= eps:
            break

        widths += addition
        remaining = float(total_width - widths.sum())
        active = (max_widths - widths) > eps

    if remaining > 1e-3:
        raise ValueError("Slope cap is too strict to allocate the full output range.")

    return widths


def compute_segment_slopes(
    knot_stack: np.ndarray,
    anchor_stack: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    spans = np.maximum(np.diff(knot_stack.astype(np.float64), axis=0), eps)
    widths = np.diff(anchor_stack.astype(np.float64), axis=0)
    return widths / spans


def _allocate_group_widths(
    spans: np.ndarray,
    base_weights: np.ndarray,
    total_width: float,
    span_lambda: float,
    eps: float,
) -> np.ndarray:
    spans = np.asarray(spans, dtype=np.float64)
    if spans.size == 0:
        return np.zeros(0, dtype=np.float64)
    if spans.size == 1:
        return np.array([float(total_width)], dtype=np.float64)

    weights = np.asarray(base_weights, dtype=np.float64)
    span_ref = float(np.median(spans))
    scores = weights * np.power((spans + eps) / (span_ref + eps), span_lambda)
    score_sum = float(scores.sum())
    if score_sum <= eps:
        return np.full(spans.shape, float(total_width) / spans.size, dtype=np.float64)
    return float(total_width) * scores / score_sum


def _allocate_linear_group_widths(
    spans: np.ndarray,
    total_width: float,
    eps: float,
) -> np.ndarray:
    spans = np.asarray(spans, dtype=np.float64)
    if spans.size == 0:
        return np.zeros(0, dtype=np.float64)
    span_sum = float(spans.sum())
    if span_sum <= eps:
        return np.full(spans.shape, float(total_width) / spans.size, dtype=np.float64)
    return float(total_width) * spans / span_sum


def _find_split_index(percentiles: tuple[float, ...], split_percentile: float, eps: float = 1e-6) -> int:
    for idx, value in enumerate(percentiles):
        if abs(float(value) - float(split_percentile)) <= eps:
            return idx
    raise ValueError(
        "tail_split_percentile must exactly match one of the configured percentiles."
    )


def compute_tail_adaptive_segment_totals(
    body_span: float,
    tail_span: float,
    total_width: float = 255.0,
    tail_suppression_gain: float = 1.0,
    tail_activation_ratio: float = 0.05,
    tail_min_output: float = 4.0,
    body_linear_cap_ratio: float | None = 1.5,
    body_tail_slope_ratio_cap: float | None = None,
    eps: float = 1e-6,
) -> tuple[float, float, dict[str, float | bool]]:
    body_span = max(float(body_span), eps)
    tail_span = max(float(tail_span), eps)
    total_span = body_span + tail_span

    linear_body_width = total_width * body_span / total_span
    linear_tail_width = total_width - linear_body_width

    tail_ratio = tail_span / body_span
    linear_cap_applied = False
    slope_ratio_cap_applied = False
    tail_adjustment_active = tail_ratio > float(0.3)  # This threshold can be tuned based on empirical observations

    if tail_adjustment_active:
        tail_excess = max(0.0, tail_ratio - float(tail_activation_ratio))
        compression = 1.0 / (1.0 + float(tail_suppression_gain) * tail_excess)

        tail_width = linear_tail_width * compression
        tail_width = float(np.clip(tail_width, float(tail_min_output), total_width))
        body_width = float(total_width - tail_width)

        if body_linear_cap_ratio is not None:
            if float(body_linear_cap_ratio) < 1.0:
                raise ValueError("body_linear_cap_ratio should be at least 1.0.")
            max_body_width = min(total_width, float(body_linear_cap_ratio) * linear_body_width)
            if body_width > max_body_width:
                body_width = float(max_body_width)
                tail_width = float(total_width - body_width)
                linear_cap_applied = True

        if body_tail_slope_ratio_cap is not None:
            if float(body_tail_slope_ratio_cap) <= 0.0:
                raise ValueError("body_tail_slope_ratio_cap should be positive.")
            max_body_by_ratio = (
                total_width
                * float(body_tail_slope_ratio_cap)
                * body_span
                / (tail_span + float(body_tail_slope_ratio_cap) * body_span)
            )
            if body_width > max_body_by_ratio:
                body_width = float(max_body_by_ratio)
                tail_width = float(total_width - body_width)
                slope_ratio_cap_applied = True
    else:
        tail_excess = 0.0
        compression = 1.0
        body_width = float(linear_body_width)
        tail_width = float(linear_tail_width)

    linear_body_slope = linear_body_width / body_span
    linear_tail_slope = linear_tail_width / tail_span
    body_slope = body_width / body_span
    tail_slope = tail_width / tail_span
    slope_ratio = body_slope / max(tail_slope, eps)
    body_slope_gain = body_slope / max(linear_body_slope, eps)
    tail_slope_gain = tail_slope / max(linear_tail_slope, eps)

    return body_width, tail_width, {
        "body_span": body_span,
        "tail_span": tail_span,
        "linear_body_width": linear_body_width,
        "linear_tail_width": linear_tail_width,
        "linear_body_slope": linear_body_slope,
        "linear_tail_slope": linear_tail_slope,
        "tail_ratio": tail_ratio,
        "tail_excess": tail_excess,
        "tail_compression": compression,
        "body_width": body_width,
        "tail_width": tail_width,
        "body_slope": body_slope,
        "tail_slope": tail_slope,
        "body_slope_gain": body_slope_gain,
        "tail_slope_gain": tail_slope_gain,
        "body_tail_slope_ratio": slope_ratio,
        "tail_adjustment_active": tail_adjustment_active,
        "linear_cap_applied": linear_cap_applied,
        "slope_ratio_cap_applied": slope_ratio_cap_applied,
    }


def build_output_anchors_from_spans(
    knot_stack: np.ndarray,
    percentiles: tuple[float, ...],
    span_lambda: float = 1.0,
    slope_cap_ratio: float | None = 4.0,
    eps: float = 1e-6,
) -> np.ndarray:
    base = np.diff(np.asarray(percentiles, dtype=np.float64))
    base = 255.0 * base / np.maximum(base.sum(), eps)

    anchors = np.zeros_like(knot_stack, dtype=np.float64)
    num_segments = knot_stack.shape[0] - 1

    for i, j in np.ndindex(knot_stack.shape[1:]):
        spans = np.maximum(np.diff(knot_stack[:, i, j]), eps)
        span_ref = float(np.median(spans))
        scores = base * np.power((spans + eps) / (span_ref + eps), span_lambda)
        widths = 255.0 * scores / np.maximum(scores.sum(), eps)

        if slope_cap_ratio is not None:
            if slope_cap_ratio < 1.0:
                raise ValueError("slope_cap_ratio should be at least 1.0.")
            total_span = max(float(knot_stack[-1, i, j] - knot_stack[0, i, j]), eps)
            linear_slope = 255.0 / total_span
            max_widths = slope_cap_ratio * linear_slope * spans
            widths = project_widths_with_caps(widths, max_widths, total_width=255.0, eps=eps)

        anchors[0, i, j] = 0.0
        anchors[1:, i, j] = np.cumsum(widths[:num_segments])

    return anchors


def build_output_anchors_tail_adaptive_split(
    knot_stack: np.ndarray,
    percentiles: tuple[float, ...],
    split_percentile: float = 99.0,
    span_lambda: float = 1.0,
    tail_suppression_gain: float = 1.0,
    tail_activation_ratio: float = 0.05,
    tail_min_output: float = 4.0,
    body_linear_cap_ratio: float | None = 1.5,
    body_tail_slope_ratio_cap: float | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    anchors = np.zeros_like(knot_stack, dtype=np.float64)
    segment_widths = np.zeros((knot_stack.shape[0] - 1, *knot_stack.shape[1:]), dtype=np.float64)

    body_span_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_span_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    linear_body_width_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    linear_tail_width_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    linear_body_slope_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    linear_tail_slope_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    body_width_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_width_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_ratio_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_excess_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_compression_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    body_slope_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_slope_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    body_slope_gain_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_slope_gain_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    slope_ratio_grid = np.zeros(knot_stack.shape[1:], dtype=np.float64)
    tail_adjustment_active_mask = np.zeros(knot_stack.shape[1:], dtype=bool)
    linear_cap_mask = np.zeros(knot_stack.shape[1:], dtype=bool)
    slope_ratio_cap_mask = np.zeros(knot_stack.shape[1:], dtype=bool)

    split_idx = _find_split_index(percentiles, split_percentile, eps=eps)
    if split_idx <= 0 or split_idx >= len(percentiles) - 1:
        raise ValueError("tail_split_percentile must create both body and tail segments.")

    for i, j in np.ndindex(knot_stack.shape[1:]):
        spans = np.maximum(np.diff(knot_stack[:, i, j]), eps)
        body_spans = spans[:split_idx]
        tail_spans = spans[split_idx:]

        body_width, tail_width, stats = compute_tail_adaptive_segment_totals(
            body_span=float(body_spans.sum()),
            tail_span=float(tail_spans.sum()),
            total_width=255.0,
            tail_suppression_gain=tail_suppression_gain,
            tail_activation_ratio=tail_activation_ratio,
            tail_min_output=tail_min_output,
            body_linear_cap_ratio=body_linear_cap_ratio,
            body_tail_slope_ratio_cap=body_tail_slope_ratio_cap,
            eps=eps,
        )

        widths = np.zeros_like(spans, dtype=np.float64)
        widths[:split_idx] = _allocate_linear_group_widths(
            body_spans,
            total_width=body_width,
            eps=eps,
        )
        widths[split_idx:] = _allocate_linear_group_widths(
            tail_spans,
            total_width=tail_width,
            eps=eps,
        )

        anchors[0, i, j] = 0.0
        anchors[1:, i, j] = np.cumsum(widths)
        segment_widths[:, i, j] = widths

        body_span_grid[i, j] = float(stats["body_span"])
        tail_span_grid[i, j] = float(stats["tail_span"])
        linear_body_width_grid[i, j] = float(stats["linear_body_width"])
        linear_tail_width_grid[i, j] = float(stats["linear_tail_width"])
        linear_body_slope_grid[i, j] = float(stats["linear_body_slope"])
        linear_tail_slope_grid[i, j] = float(stats["linear_tail_slope"])
        body_width_grid[i, j] = float(stats["body_width"])
        tail_width_grid[i, j] = float(stats["tail_width"])
        tail_ratio_grid[i, j] = float(stats["tail_ratio"])
        tail_excess_grid[i, j] = float(stats["tail_excess"])
        tail_compression_grid[i, j] = float(stats["tail_compression"])
        body_slope_grid[i, j] = float(stats["body_slope"])
        tail_slope_grid[i, j] = float(stats["tail_slope"])
        body_slope_gain_grid[i, j] = float(stats["body_slope_gain"])
        tail_slope_gain_grid[i, j] = float(stats["tail_slope_gain"])
        slope_ratio_grid[i, j] = float(stats["body_tail_slope_ratio"])
        tail_adjustment_active_mask[i, j] = bool(stats["tail_adjustment_active"])
        linear_cap_mask[i, j] = bool(stats["linear_cap_applied"])
        slope_ratio_cap_mask[i, j] = bool(stats["slope_ratio_cap_applied"])

    return anchors, {
        "segment_widths": segment_widths,
        "segment_slopes": compute_segment_slopes(knot_stack, anchors, eps=eps),
        "body_span_grid": body_span_grid,
        "tail_span_grid": tail_span_grid,
        "linear_body_width_grid": linear_body_width_grid,
        "linear_tail_width_grid": linear_tail_width_grid,
        "linear_body_slope_grid": linear_body_slope_grid,
        "linear_tail_slope_grid": linear_tail_slope_grid,
        "body_output_width_grid": body_width_grid,
        "tail_output_width_grid": tail_width_grid,
        "tail_ratio_grid": tail_ratio_grid,
        "tail_excess_grid": tail_excess_grid,
        "tail_compression_grid": tail_compression_grid,
        "body_slope_grid": body_slope_grid,
        "tail_slope_grid": tail_slope_grid,
        "body_slope_gain_grid": body_slope_gain_grid,
        "tail_slope_gain_grid": tail_slope_gain_grid,
        "body_tail_slope_ratio_grid": slope_ratio_grid,
        "tail_adjustment_active_mask": tail_adjustment_active_mask,
        "linear_cap_mask": linear_cap_mask,
        "slope_ratio_cap_mask": slope_ratio_cap_mask,
        "split_index": np.full(knot_stack.shape[1:], split_idx, dtype=np.int32),
    }


def build_output_anchors(
    knot_stack: np.ndarray,
    percentiles: tuple[float, ...],
    anchor_mode: Literal["span", "tail_adaptive_split"] = "span",
    span_lambda: float = 1.0,
    slope_cap_ratio: float | None = 4.0,
    tail_split_percentile: float = 99.0,
    tail_suppression_gain: float = 1.0,
    tail_activation_ratio: float = 0.05,
    tail_min_output: float = 4.0,
    body_linear_cap_ratio: float | None = 1.5,
    body_tail_slope_ratio_cap: float | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray | str | float | None]]:
    if anchor_mode == "span":
        anchors = build_output_anchors_from_spans(
            knot_stack=knot_stack,
            percentiles=percentiles,
            span_lambda=span_lambda,
            slope_cap_ratio=slope_cap_ratio,
            eps=eps,
        )
        debug: dict[str, np.ndarray | str | float | None] = {
            "anchor_mode": anchor_mode,
            "segment_widths": np.diff(anchors, axis=0).astype(np.float64),
            "segment_slopes": compute_segment_slopes(knot_stack, anchors, eps=eps),
            "body_span_grid": None,
            "tail_span_grid": None,
            "linear_body_width_grid": None,
            "linear_tail_width_grid": None,
            "linear_body_slope_grid": None,
            "linear_tail_slope_grid": None,
            "body_output_width_grid": None,
            "tail_output_width_grid": None,
            "tail_ratio_grid": None,
            "tail_excess_grid": None,
            "tail_compression_grid": None,
            "body_slope_grid": None,
            "tail_slope_grid": None,
            "body_slope_gain_grid": None,
            "tail_slope_gain_grid": None,
            "body_tail_slope_ratio_grid": None,
            "tail_adjustment_active_mask": None,
            "linear_cap_mask": None,
            "slope_ratio_cap_mask": None,
            "tail_split_percentile": None,
        }
        return anchors, debug

    if anchor_mode == "tail_adaptive_split":
        anchors, debug = build_output_anchors_tail_adaptive_split(
            knot_stack=knot_stack,
            percentiles=percentiles,
            split_percentile=tail_split_percentile,
            span_lambda=span_lambda,
            tail_suppression_gain=tail_suppression_gain,
            tail_activation_ratio=tail_activation_ratio,
            tail_min_output=tail_min_output,
            body_linear_cap_ratio=body_linear_cap_ratio,
            body_tail_slope_ratio_cap=body_tail_slope_ratio_cap,
            eps=eps,
        )
        debug["anchor_mode"] = anchor_mode
        debug["tail_split_percentile"] = float(tail_split_percentile)
        return anchors, debug

    raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def resize_stack(stack: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    resized = [
        cv2.resize(plane.astype(np.float64), dsize=(w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        for plane in stack
    ]
    return np.stack(resized, axis=0)


def piecewise_linear_rescale(
    image: np.ndarray,
    knot_fields: np.ndarray,
    anchor_fields: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    if knot_fields.shape != anchor_fields.shape:
        raise ValueError("knot_fields and anchor_fields must have the same shape.")
    if knot_fields.shape[1:] != image.shape:
        raise ValueError("Field shape must match the input image shape.")

    knots = knot_fields.astype(np.float64, copy=True)
    for idx in range(1, knots.shape[0]):
        knots[idx] = np.maximum(knots[idx], knots[idx - 1] + eps)

    img = image.astype(np.float64)
    img_clip = np.clip(img, knots[0], knots[-1])
    out = np.zeros_like(img_clip, dtype=np.float64)

    for idx in range(knots.shape[0] - 1):
        x0 = knots[idx]
        x1 = knots[idx + 1]
        y0 = anchor_fields[idx]
        y1 = anchor_fields[idx + 1]
        if idx == knots.shape[0] - 2:
            mask = img_clip >= x0
        else:
            mask = (img_clip >= x0) & (img_clip < x1)

        den = np.maximum(x1 - x0, eps)
        t = (img_clip - x0) / den
        out[mask] = y0[mask] + t[mask] * (y1[mask] - y0[mask])

    return np.clip(out, 0, 255).astype(np.uint8)


def sample_grid_values(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    samples_per_grid: int = 96,
) -> dict[tuple[int, int], np.ndarray]:
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output: dict[tuple[int, int], np.ndarray] = {}

    for i, j in np.ndindex(grid_shape):
        patch = image[
            patch_shape[0] * i: patch_shape[0] * (i + 1),
            patch_shape[1] * j: patch_shape[1] * (j + 1),
        ].astype(np.float64).reshape(-1)
        patch_sorted = np.sort(patch)
        if patch_sorted.size <= samples_per_grid:
            output[(i, j)] = patch_sorted
            continue

        indices = np.linspace(0, patch_sorted.size - 1, samples_per_grid).round().astype(int)
        output[(i, j)] = patch_sorted[indices]

    return output


def evaluate_piecewise_curve(
    raw_values: np.ndarray,
    knot_x: np.ndarray,
    knot_y: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(raw_values.astype(np.float64), float(knot_x[0]), float(knot_x[-1]))
    output = np.zeros_like(clipped, dtype=np.float64)

    for idx in range(len(knot_x) - 1):
        x0 = float(knot_x[idx])
        x1 = max(float(knot_x[idx + 1]), x0 + eps)
        y0 = float(knot_y[idx])
        y1 = float(knot_y[idx + 1])

        if idx == len(knot_x) - 2:
            mask = clipped >= x0
        else:
            mask = (clipped >= x0) & (clipped < x1)

        t = (clipped[mask] - x0) / (x1 - x0)
        output[mask] = y0 + t * (y1 - y0)

    return clipped, output


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image_f64 = image.astype(np.float64, copy=False)
    vmin = float(np.min(image_f64))
    vmax = float(np.max(image_f64))
    if vmax <= vmin:
        return np.zeros_like(image_f64, dtype=np.uint8)
    scaled = 255.0 * (image_f64 - vmin) / (vmax - vmin)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _to_bgr_preview(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        base = image if image.dtype == np.uint8 else _normalize_to_uint8(image)
        return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=True)
    raise ValueError("Expected a grayscale image or a BGR image.")


def _blend_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    roi = image[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    fill = np.empty_like(roi, dtype=np.float64)
    fill[...] = np.asarray(color, dtype=np.float64)
    blended = (1.0 - alpha) * roi + alpha * fill
    image[y0:y1 + 1, x0:x1 + 1] = np.clip(blended, 0, 255).astype(np.uint8)


def build_side_by_side_preview(
    left_image: np.ndarray,
    right_image: np.ndarray,
    gap: int = 10,
    margin: int = 8,
    bg_color: tuple[int, int, int] = (18, 18, 18),
    left_label: str = "rescaled",
    right_label: str = "overlay",
) -> np.ndarray:
    left = _to_bgr_preview(left_image)
    right = _to_bgr_preview(right_image)

    if left.shape[0] != right.shape[0]:
        raise ValueError("left_image and right_image must have the same height.")

    height = left.shape[0]
    width = left.shape[1] + right.shape[1] + gap + 2 * margin
    title_band = 28

    canvas = np.full((height + title_band + 2 * margin, width, 3), bg_color, dtype=np.uint8)

    left_x = margin
    right_x = margin + left.shape[1] + gap
    y = margin + title_band

    canvas[y:y + left.shape[0], left_x:left_x + left.shape[1]] = left
    canvas[y:y + right.shape[0], right_x:right_x + right.shape[1]] = right

    cv2.putText(
        canvas,
        left_label,
        (left_x, margin + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        right_label,
        (right_x, margin + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (left_x - 1, y - 1), (left_x + left.shape[1], y + left.shape[0]), (80, 80, 80), 1)
    cv2.rectangle(canvas, (right_x - 1, y - 1), (right_x + right.shape[1], y + right.shape[0]), (80, 80, 80), 1)

    return canvas


def _grid_cell_bounds(
    image_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    row: int,
    col: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    rows, cols = grid_shape
    cell_h = height // rows
    cell_w = width // cols
    y0 = row * cell_h
    y1 = height if row == rows - 1 else (row + 1) * cell_h
    x0 = col * cell_w
    x1 = width if col == cols - 1 else (col + 1) * cell_w
    return x0, y0, x1, y1


def _draw_grid_lines(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    rows, cols = grid_shape
    height, width = image.shape[:2]
    cell_h = height // rows
    cell_w = width // cols

    for row in range(1, rows):
        y = row * cell_h
        cv2.line(image, (0, y), (width - 1, y), color, thickness=thickness, lineType=cv2.LINE_AA)
    for col in range(1, cols):
        x = col * cell_w
        cv2.line(image, (x, 0), (x, height - 1), color, thickness=thickness, lineType=cv2.LINE_AA)


def _format_debug_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    if not np.isfinite(float(value)):
        return "n/a"
    return fmt.format(float(value))


def build_grid_value_overlay(
    image: np.ndarray,
    value_grid: np.ndarray | None,
    title: str,
    value_format: str = "{:.3f}",
    active_mask: np.ndarray | None = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    overlay = _to_bgr_preview(image)
    height, width = overlay.shape[:2]

    if value_grid is None:
        _blend_rect(overlay, 8, 8, min(width - 9, 250), min(height - 9, 30), (18, 18, 18), 0.82)
        cv2.putText(
            overlay,
            f"{title}: n/a",
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return overlay

    rows, cols = value_grid.shape
    finite_values = value_grid[np.isfinite(value_grid)]
    value_min = float(np.min(finite_values)) if finite_values.size else 0.0
    value_max = float(np.max(finite_values)) if finite_values.size else 1.0
    if value_max <= value_min:
        value_max = value_min + 1.0

    normalized = np.clip(255.0 * (value_grid - value_min) / (value_max - value_min), 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, colormap)
    heatmap = cv2.resize(heatmap, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(overlay, 0.42, heatmap, 0.58, 0.0)

    _blend_rect(overlay, 8, 8, min(width - 9, 340), min(height - 9, 32), (18, 18, 18), 0.82)
    cv2.putText(
        overlay,
        f"{title}  min {value_min:.3f}  max {value_max:.3f}",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    for row, col in np.ndindex((rows, cols)):
        x0, y0, x1, y1 = _grid_cell_bounds((height, width), (rows, cols), row, col)
        label_w = min(x1 - x0 - 2, 64)
        label_h = min(y1 - y0 - 2, 18)
        if label_w > 8 and label_h > 8:
            _blend_rect(
                overlay,
                x0 + 2,
                y0 + 2,
                x0 + label_w,
                y0 + label_h,
                (18, 18, 18),
                0.76,
            )
            cv2.putText(
                overlay,
                _format_debug_value(float(value_grid[row, col]), value_format),
                (x0 + 5, y0 + min(label_h - 3, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        if active_mask is not None and bool(active_mask[row, col]):
            cv2.rectangle(
                overlay,
                (x0 + 1, y0 + 1),
                (x1 - 2, y1 - 2),
                (70, 255, 120),
                thickness=2,
            )

    _draw_grid_lines(overlay, (rows, cols))
    return overlay


def build_grid_choice_overlay(
    image: np.ndarray,
    choice_grid: np.ndarray | None,
    title: str = "p99 fix choice",
) -> np.ndarray:
    overlay = _to_bgr_preview(image)
    height, width = overlay.shape[:2]

    if choice_grid is None:
        _blend_rect(overlay, 8, 8, min(width - 9, 260), min(height - 9, 30), (18, 18, 18), 0.82)
        cv2.putText(
            overlay,
            f"{title}: n/a",
            (14, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return overlay

    rows, cols = choice_grid.shape
    legend = [
        ("keep", (70, 70, 70)),
        ("raw98", (0, 170, 255)),
        ("les-1", (255, 120, 80)),
    ]
    choice_labels = {0: "k", 1: "r98", 2: "l-1"}
    choice_colors = {
        0: (70, 70, 70),
        1: (0, 170, 255),
        2: (255, 120, 80),
    }

    _blend_rect(overlay, 8, 8, min(width - 9, 360), min(height - 9, 36), (18, 18, 18), 0.84)
    cv2.putText(
        overlay,
        title,
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    legend_x = 98
    for label, color in legend:
        cv2.rectangle(overlay, (legend_x, 12), (legend_x + 12, 24), color, thickness=-1)
        cv2.rectangle(overlay, (legend_x, 12), (legend_x + 12, 24), (230, 230, 230), thickness=1)
        cv2.putText(
            overlay,
            label,
            (legend_x + 18, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        legend_x += 76

    for row, col in np.ndindex((rows, cols)):
        x0, y0, x1, y1 = _grid_cell_bounds((height, width), (rows, cols), row, col)
        choice_code = int(choice_grid[row, col])
        _blend_rect(
            overlay,
            x0 + 1,
            y0 + 1,
            x1 - 2,
            y1 - 2,
            choice_colors.get(choice_code, (70, 70, 70)),
            0.44,
        )
        cv2.putText(
            overlay,
            choice_labels.get(choice_code, "?"),
            (x0 + 6, y0 + min(18, max(12, (y1 - y0) // 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    _draw_grid_lines(overlay, (rows, cols))
    return overlay


def _find_percentile_index_or_none(
    percentiles: tuple[float, ...] | list[float],
    target: float,
    eps: float = 1e-6,
) -> int | None:
    for idx, value in enumerate(percentiles):
        if abs(float(value) - float(target)) <= eps:
            return idx
    return None


def build_grid_distribution_overlay(
    image: np.ndarray,
    raw_image: np.ndarray,
    grid_shape: tuple[int, int],
    knot_grids: np.ndarray | None = None,
    percentiles: tuple[float, ...] | list[float] | None = None,
    bins: int = 24,
    panel_alpha: float = 0.84,
    panel_color: tuple[int, int, int] = (18, 18, 18),
    bar_color: tuple[int, int, int] = (120, 210, 255),
    axis_color: tuple[int, int, int] = (210, 210, 210),
    border_color: tuple[int, int, int] = (245, 245, 245),
    grid_color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    overlay = _to_bgr_preview(image)
    raw_values = raw_image.astype(np.float64, copy=False)
    rows, cols = grid_shape
    height, width = overlay.shape[:2]
    marker_specs = [
        (5.0, (255, 255, 0)),
        (95.0, (0, 255, 255)),
        (99.0, (0, 170, 255)),
        (100.0, (0, 0, 255)),
    ]

    for row, col in np.ndindex((rows, cols)):
        x0, y0, x1, y1 = _grid_cell_bounds((height, width), (rows, cols), row, col)
        patch = raw_values[y0:y1, x0:x1].reshape(-1)
        patch_min = float(np.min(patch))
        patch_max = float(np.max(patch))
        if patch_max <= patch_min:
            patch_max = patch_min + 1.0

        pad_x = max(4, int(round((x1 - x0) * 0.10)))
        pad_y = max(4, int(round((y1 - y0) * 0.12)))
        label_band_h = max(14, int(round((y1 - y0) * 0.16)))
        chart_x0 = min(max(x0 + pad_x, x0), x1 - 2)
        chart_x1 = max(min(x1 - pad_x, x1 - 1), chart_x0 + 1)
        chart_y0 = min(max(y0 + pad_y, y0), y1 - 2)
        chart_y1 = max(min(y1 - pad_y - label_band_h, y1 - 1), chart_y0 + 8)

        _blend_rect(overlay, chart_x0, chart_y0, chart_x1, chart_y1, panel_color, panel_alpha)
        cv2.rectangle(overlay, (chart_x0, chart_y0), (chart_x1, chart_y1), border_color, thickness=1)

        hist, edges = np.histogram(patch, bins=max(4, int(bins)), range=(patch_min, patch_max))
        hist = hist.astype(np.float64)
        hist_max = float(np.max(hist)) if hist.size else 1.0
        if hist_max <= 0.0:
            hist_max = 1.0

        mid_y = chart_y0 + (chart_y1 - chart_y0) // 2
        mid_x = chart_x0 + (chart_x1 - chart_x0) // 2
        cv2.line(overlay, (chart_x0, mid_y), (chart_x1, mid_y), grid_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (mid_x, chart_y0), (mid_x, chart_y1), grid_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (chart_x0, chart_y1), (chart_x1, chart_y1), axis_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (chart_x0, chart_y0), (chart_x0, chart_y1), axis_color, thickness=1, lineType=cv2.LINE_AA)

        bar_count = max(1, hist.size)
        bar_w = max(1, (chart_x1 - chart_x0 + 1) // bar_count)
        for idx, bin_value in enumerate(hist):
            bx0 = chart_x0 + idx * bar_w
            bx1 = chart_x1 if idx == bar_count - 1 else min(chart_x1, bx0 + bar_w - 1)
            bar_h = int(round((bin_value / hist_max) * max(1, chart_y1 - chart_y0 - 1)))
            by0 = max(chart_y0, chart_y1 - bar_h)
            cv2.rectangle(overlay, (bx0, by0), (bx1, chart_y1), bar_color, thickness=-1)

        if knot_grids is not None and percentiles is not None:
            for marker_percentile, marker_color in marker_specs:
                marker_idx = _find_percentile_index_or_none(percentiles, marker_percentile)
                if marker_idx is None or marker_idx >= knot_grids.shape[0]:
                    continue
                knot_value = float(knot_grids[marker_idx, row, col])
                px = chart_x0 + int(round((knot_value - patch_min) / (patch_max - patch_min) * (chart_x1 - chart_x0)))
                px = int(np.clip(px, chart_x0, chart_x1))
                cv2.line(overlay, (px, chart_y0), (px, chart_y1), marker_color, thickness=1, lineType=cv2.LINE_AA)

        left_label = f"{patch_min:.0f}"
        right_label = f"{patch_max:.0f}"
        left_size, _ = cv2.getTextSize(left_label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
        right_size, _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
        label_y0 = min(max(chart_y1 + 2, y0 + 2), y1 - 12)
        label_y1 = min(y1 - 2, label_y0 + 12)
        left_box_x1 = min(chart_x1 - 2, chart_x0 + left_size[0] + 8)
        right_box_x0 = max(chart_x0 + 2, chart_x1 - right_size[0] - 8)

        _blend_rect(overlay, chart_x0 + 2, label_y0, left_box_x1, label_y1, (18, 18, 18), 0.92)
        _blend_rect(overlay, right_box_x0, label_y0, chart_x1 - 2, label_y1, (18, 18, 18), 0.92)
        cv2.putText(
            overlay,
            left_label,
            (chart_x0 + 4, label_y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            right_label,
            (right_box_x0 + 2, label_y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    _draw_grid_lines(overlay, (rows, cols))
    return overlay


def build_grid_curve_overlay(
    image: np.ndarray,
    knot_grids: np.ndarray,
    anchor_grids: np.ndarray,
    raw_image: np.ndarray | None = None,
    raw_samples_per_grid: int = 96,
    panel_alpha: float = 0.82,
    panel_color: tuple[int, int, int] = (18, 18, 18),
    line_color: tuple[int, int, int] = (40, 255, 120),
    point_color: tuple[int, int, int] = (20, 210, 255),
    raw_point_color: tuple[int, int, int] = (80, 220, 255),
    axis_color: tuple[int, int, int] = (210, 210, 210),
    border_color: tuple[int, int, int] = (245, 245, 245),
    grid_color: tuple[int, int, int] = (120, 120, 120),
) -> np.ndarray:
    if knot_grids.shape != anchor_grids.shape:
        raise ValueError("knot_grids and anchor_grids must have the same shape.")

    rows, cols = knot_grids.shape[1:]
    height, width = image.shape[:2]
    cell_h = height // rows
    cell_w = width // cols

    overlay = _to_bgr_preview(image)
    raw_samples_by_grid = None
    if raw_image is not None:
        if raw_image.shape[:2] != image.shape[:2]:
            raise ValueError("raw_image must have the same height and width as image.")
        raw_samples_by_grid = sample_grid_values(raw_image, (rows, cols), samples_per_grid=raw_samples_per_grid)

    for row, col in np.ndindex((rows, cols)):
        y0 = row * cell_h
        y1 = height if row == rows - 1 else (row + 1) * cell_h
        x0 = col * cell_w
        x1 = width if col == cols - 1 else (col + 1) * cell_w

        pad_x = max(4, int(round((x1 - x0) * 0.10)))
        pad_y = max(4, int(round((y1 - y0) * 0.12)))
        chart_x0 = min(max(x0 + pad_x, x0), x1 - 2)
        chart_x1 = max(min(x1 - pad_x, x1 - 1), chart_x0 + 1)
        chart_y0 = min(max(y0 + pad_y, y0), y1 - 2)
        chart_y1 = max(min(y1 - pad_y, y1 - 1), chart_y0 + 1)

        _blend_rect(overlay, chart_x0, chart_y0, chart_x1, chart_y1, panel_color, panel_alpha)
        cv2.rectangle(overlay, (chart_x0, chart_y0), (chart_x1, chart_y1), border_color, thickness=1)

        knots = knot_grids[:, row, col].astype(np.float64)
        anchors = anchor_grids[:, row, col].astype(np.float64)
        x_min = float(knots[0])
        x_max = float(knots[-1])
        if x_max <= x_min:
            x_max = x_min + 1.0

        points = []
        for knot, anchor in zip(knots, anchors):
            px = chart_x0 + int(round((knot - x_min) / (x_max - x_min) * (chart_x1 - chart_x0)))
            py = chart_y1 - int(round(anchor / 255.0 * (chart_y1 - chart_y0)))
            px = int(np.clip(px, chart_x0, chart_x1))
            py = int(np.clip(py, chart_y0, chart_y1))
            points.append((px, py))

        mid_y = chart_y0 + (chart_y1 - chart_y0) // 2
        mid_x = chart_x0 + (chart_x1 - chart_x0) // 2
        cv2.line(overlay, (chart_x0, mid_y), (chart_x1, mid_y), grid_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (mid_x, chart_y0), (mid_x, chart_y1), grid_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (chart_x0, chart_y1), (chart_x1, chart_y1), axis_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (chart_x0, chart_y0), (chart_x0, chart_y1), axis_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(
            overlay,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=border_color,
            thickness=4,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            overlay,
            [np.array(points, dtype=np.int32)],
            isClosed=False,
            color=line_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        if raw_samples_by_grid is not None:
            raw_values = raw_samples_by_grid[(row, col)]
            raw_x, raw_y = evaluate_piecewise_curve(raw_values, knots, anchors)
            for sample_x, sample_y in zip(raw_x, raw_y):
                px = chart_x0 + int(round((sample_x - x_min) / (x_max - x_min) * (chart_x1 - chart_x0)))
                py = chart_y1 - int(round(sample_y / 255.0 * (chart_y1 - chart_y0)))
                px = int(np.clip(px, chart_x0, chart_x1))
                py = int(np.clip(py, chart_y0, chart_y1))
                cv2.circle(overlay, (px, py), radius=1, color=raw_point_color, thickness=-1, lineType=cv2.LINE_AA)
        for px, py in points:
            cv2.circle(overlay, (px, py), radius=4, color=border_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (px, py), radius=2, color=point_color, thickness=-1, lineType=cv2.LINE_AA)

    for row in range(1, rows):
        y = row * cell_h
        cv2.line(overlay, (0, y), (width - 1, y), border_color, thickness=1, lineType=cv2.LINE_AA)
    for col in range(1, cols):
        x = col * cell_w
        cv2.line(overlay, (x, 0), (x, height - 1), border_color, thickness=1, lineType=cv2.LINE_AA)

    return overlay


def build_grid_debug_board(
    image_shape: tuple[int, int],
    anchor_mode: str,
    tail_ratio_grid: np.ndarray | None,
    body_span_grid: np.ndarray | None,
    tail_span_grid: np.ndarray | None,
    body_output_width_grid: np.ndarray | None,
    tail_output_width_grid: np.ndarray | None,
    tail_compression_grid: np.ndarray | None,
    body_slope_gain_grid: np.ndarray | None,
    tail_adjustment_active_mask: np.ndarray | None,
    linear_cap_mask: np.ndarray | None,
    p99_choice_grid: np.ndarray | None,
    frame_grid_span_median: float | None,
    frame_les_diff: float | None,
) -> np.ndarray:
    height, width = image_shape
    board = np.full((height, width, 3), (16, 16, 16), dtype=np.uint8)
    header_h = min(42, max(32, height // 10))
    rows = 1
    cols = 1

    for grid in (
        tail_ratio_grid,
        body_span_grid,
        tail_span_grid,
        body_output_width_grid,
        tail_output_width_grid,
        tail_compression_grid,
    ):
        if grid is not None:
            rows, cols = grid.shape
            break

    _blend_rect(board, 0, 0, width - 1, header_h - 1, (28, 28, 28), 0.95)
    header_text = (
        f"mode {anchor_mode} | frame p95-p5 med {_format_debug_value(frame_grid_span_median, '{:.1f}')}"
        f" | les diff {_format_debug_value(frame_les_diff, '{:.0f}')}"
    )
    cv2.putText(
        board,
        header_text,
        (12, min(header_h - 10, 26)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    cell_h = max(1, (height - header_h) // rows)
    cell_w = max(1, width // cols)

    for row, col in np.ndindex((rows, cols)):
        x0 = col * cell_w
        x1 = width if col == cols - 1 else (col + 1) * cell_w
        y0 = header_h + row * cell_h
        y1 = height if row == rows - 1 else header_h + (row + 1) * cell_h

        base_color = (46, 46, 46)
        if tail_adjustment_active_mask is not None and bool(tail_adjustment_active_mask[row, col]):
            base_color = (36, 70, 44)
        _blend_rect(board, x0, y0, x1 - 1, y1 - 1, base_color, 0.26)
        cv2.rectangle(board, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 90), 1)

        if linear_cap_mask is not None and bool(linear_cap_mask[row, col]):
            cv2.rectangle(board, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (0, 170, 255), 2)

        p99_choice_text = "n/a"
        if p99_choice_grid is not None:
            p99_choice_text = {
                0: "keep",
                1: "r98",
                2: "l-1",
            }.get(int(p99_choice_grid[row, col]), "?")

        lines = [
            f"tr {_format_debug_value(None if tail_ratio_grid is None else float(tail_ratio_grid[row, col]), '{:.3f}')} "
            f"bg x{_format_debug_value(None if body_slope_gain_grid is None else float(body_slope_gain_grid[row, col]), '{:.2f}')}",
            f"span {_format_debug_value(None if body_span_grid is None else float(body_span_grid[row, col]), '{:.0f}')}|"
            f"{_format_debug_value(None if tail_span_grid is None else float(tail_span_grid[row, col]), '{:.0f}')}",
            f"out {_format_debug_value(None if body_output_width_grid is None else float(body_output_width_grid[row, col]), '{:.0f}')}|"
            f"{_format_debug_value(None if tail_output_width_grid is None else float(tail_output_width_grid[row, col]), '{:.0f}')}",
            f"cmp {_format_debug_value(None if tail_compression_grid is None else float(tail_compression_grid[row, col]), '{:.2f}')} 99 {p99_choice_text}",
            f"a {int(bool(tail_adjustment_active_mask[row, col])) if tail_adjustment_active_mask is not None else 0}"
            f" c {int(bool(linear_cap_mask[row, col])) if linear_cap_mask is not None else 0}",
        ]

        line_y = y0 + 13
        for text in lines:
            if line_y >= y1 - 2:
                break
            cv2.putText(
                board,
                text,
                (x0 + 4, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            line_y += 11

    return board


def build_debug_preview(
    raw_image: np.ndarray,
    rescaled_image: np.ndarray,
    curve_overlay: np.ndarray,
    distribution_overlay: np.ndarray,
    tail_ratio_overlay: np.ndarray,
    p99_choice_overlay: np.ndarray,
    body_slope_gain_overlay: np.ndarray,
) -> np.ndarray:
    panels = [
        _to_bgr_preview(raw_image),
        _to_bgr_preview(rescaled_image),
        _to_bgr_preview(curve_overlay),
        _to_bgr_preview(distribution_overlay),
        _to_bgr_preview(tail_ratio_overlay),
        _to_bgr_preview(p99_choice_overlay),
        _to_bgr_preview(body_slope_gain_overlay),
    ]
    labels = ["raw", "rescaled", "curve", "distribution", "tail ratio", "p99 choice", "body slope x"]
    panel_h, panel_w = panels[0].shape[:2]
    gap = 8
    margin = 8
    num_cols = 3
    num_rows = int(np.ceil(len(panels) / num_cols))
    canvas = np.full(
        (num_rows * panel_h + (num_rows - 1) * gap + 2 * margin, num_cols * panel_w + (num_cols - 1) * gap + 2 * margin, 3),
        (18, 18, 18),
        dtype=np.uint8,
    )

    positions = []
    for idx in range(len(panels)):
        row = idx // num_cols
        col = idx % num_cols
        positions.append((margin + col * (panel_w + gap), margin + row * (panel_h + gap)))

    for (x, y), panel, label in zip(positions, panels, labels):
        canvas[y:y + panel_h, x:x + panel_w] = panel
        cv2.rectangle(canvas, (x - 1, y - 1), (x + panel_w, y + panel_h), (90, 90, 90), 1)
        _blend_rect(canvas, x + 4, y + 4, min(x + 116, x + panel_w - 4), min(y + 24, y + panel_h - 4), (18, 18, 18), 0.80)
        cv2.putText(
            canvas,
            label,
            (x + 9, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    return canvas


class MultiKnotPercentileFieldscale(Fieldscale):
    def __init__(
        self,
        max_diff: float = 400,
        min_diff: float = 400,
        iteration: int = 7,
        gamma: float = 1.5,
        clahe: bool = True,
        video: bool = False,
        ema_period: int = 10,
        ema_alpha_override: float | None = None,
        les_mode: Literal["single", "repeat5"] = "single",
        grid_shape: tuple[int, int] = (8, 8),
        percentiles: Iterable[float] = (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100),
        mid_alpha: float = 0.5,
        min_interval: float = 1.0,
        anchor_mode: Literal["span", "tail_adaptive_split"] = "span",
        span_lambda: float = 1.0,
        slope_cap_ratio: float | None = 4.0,
        tail_split_percentile: float = 99.0,
        tail_suppression_gain: float = 1.0,
        tail_activation_ratio: float = 0.05,
        tail_min_output: float = 4.0,
        body_linear_cap_ratio: float | None = 1.5,
        body_tail_slope_ratio_cap: float | None = None,
        use_les: bool = False,
        les_p95_floor: bool = True,
        p99_violation_strategy: Literal["raw98_or_les_minus_one", "les_minus_one"] = "raw98_or_les_minus_one",
        use_message_passing: bool = False,
        global_percentiles: bool = False,
        enforce_mode: Literal["bottom_up", "top_down"] = "bottom_up",
    ):
        super().__init__(
            max_diff=max_diff,
            min_diff=min_diff,
            iteration=iteration,
            gamma=gamma,
            clahe=clahe,
            video=video,
            ema_period=ema_period,
            ema_alpha_override=ema_alpha_override,
            les_mode=les_mode,
            grid_shape=grid_shape,
        )
        self.percentiles = parse_percentiles(percentiles)
        self.mid_alpha = float(mid_alpha)
        self.min_interval = float(min_interval)
        self.anchor_mode = str(anchor_mode)
        self.span_lambda = float(span_lambda)
        self.slope_cap_ratio = None if slope_cap_ratio is None else float(slope_cap_ratio)
        self.tail_split_percentile = float(tail_split_percentile)
        self.tail_suppression_gain = float(tail_suppression_gain)
        self.tail_activation_ratio = float(tail_activation_ratio)
        self.tail_min_output = float(tail_min_output)
        self.body_linear_cap_ratio = None if body_linear_cap_ratio is None else float(body_linear_cap_ratio)
        self.body_tail_slope_ratio_cap = None if body_tail_slope_ratio_cap is None else float(body_tail_slope_ratio_cap)
        self.use_les = bool(use_les)
        self.les_p95_floor = bool(les_p95_floor)
        self.p99_violation_strategy = str(p99_violation_strategy)
        self.use_message_passing = bool(use_message_passing)
        self.global_percentiles = bool(global_percentiles)
        self.enforce_mode = str(enforce_mode)
        self.prev_knot_grids = None
        self.knot_grid_buffer = deque(maxlen=self.ema_period)

    def reset_temporal_state(self) -> None:
        super().reset_temporal_state()
        self.prev_knot_grids = None
        self.knot_grid_buffer.clear()

    def _apply_knot_temporal_smoothing(
        self,
        knot_grids: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        raw_knot_grids = knot_grids.astype(np.float64, copy=True)
        prev_knot_before = None if self.prev_knot_grids is None else self.prev_knot_grids.astype(np.float64, copy=True)
        buffer_snapshot = None

        stage = "disabled"

        if self.video:
            self.knot_grid_buffer.append(raw_knot_grids.copy())
            buffer_snapshot = np.stack(list(self.knot_grid_buffer), axis=0).astype(np.float64, copy=True)

            if self.prev_knot_grids is None:
                knot_grids = np.mean(np.stack(self.knot_grid_buffer, axis=0), axis=0)
                if len(self.knot_grid_buffer) < self.ema_period:
                    stage = "warmup_sma"
                else:
                    stage = "sma_init"
                    self.prev_knot_grids = knot_grids.copy()
            else:
                stage = "ema"
                alpha = self.ema_alpha
                knot_grids = alpha * raw_knot_grids + (1.0 - alpha) * self.prev_knot_grids
                self.prev_knot_grids = knot_grids.copy()
        else:
            knot_grids = raw_knot_grids
            self.prev_knot_grids = knot_grids.copy()

        post_temporal_before_enforce = knot_grids.astype(np.float64, copy=True)
        knot_grids = enforce_knot_order(
            knot_grids,
            min_interval=self.min_interval,
            mode=self.enforce_mode,
        )

        debug = {
            "ema_stage": stage,
            "ema_alpha": self.ema_alpha if self.video else None,
            "ema_alpha_override": self.ema_alpha_override if self.video else None,
            "ema_period": self.ema_period,
            "knot_buffer_length": len(self.knot_grid_buffer),
            "prev_knot_grids_before": prev_knot_before,
            "knot_grid_buffer_snapshot": buffer_snapshot,
            "pre_ema_knot_grids": raw_knot_grids,
            "post_temporal_knot_grids_before_enforce": post_temporal_before_enforce,
            "post_ema_knot_grids": knot_grids.astype(np.float64, copy=True),
        }
        return knot_grids.astype(np.float64, copy=True), debug

    def _build_knot_grids(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        mapping_grid_shape = (1, 1) if self.global_percentiles else self.grid_shape
        knots = gridwise_percentiles(image, self.percentiles, mapping_grid_shape)
        frame_grid_span_median = None
        frame_les_diff = None
        raw_95_grid = None
        raw_98_grid = None
        les_100_before_floor_grid = None
        les_100_after_floor_grid = None
        p99_pre_adjust_grid = None
        p99_post_adjust_grid = None
        p99_violation_mask = None
        p99_choice_grid = None
        p99_replacement_grid = None
        p99_les_minus_one_grid = None
        p99_choose_raw98_mask = None
        p99_choose_les_minus_one_mask = None

        if self.use_les:
            frame_grid_span_median = compute_frame_grid_span_median(
                image=image,
                grid_shape=self.grid_shape,
                lower_percentile=5.0,
                upper_percentile=95.0,
            )
            frame_les_diff = float(self.max_diff)
            percentile_99_idx = None
            for idx, value in enumerate(self.percentiles):
                if abs(float(value) - 99.0) <= 1e-6:
                    percentile_99_idx = idx
                    break

            raw_95_grid, raw_98_grid = gridwise_percentiles(
                image,
                percentiles=(95.0, 98.0),
                grid_shape=mapping_grid_shape,
            ).astype(np.float64)

            if percentile_99_idx is not None:
                p99_pre_adjust_grid = knots[percentile_99_idx].astype(np.float64, copy=True)

            knots[-1] = self._apply_les(
                knots[-1].astype(np.float64),
                diff_threshold=frame_les_diff,
                extrema="max",
            ).astype(np.float64)
            knots[-1] = self._apply_les(
                knots[-1].astype(np.float64),
                diff_threshold=frame_les_diff,
                extrema="min",
            ).astype(np.float64)
            les_100_before_floor_grid = knots[-1].astype(np.float64, copy=True)
            if self.les_p95_floor:
                knots[-1] = np.maximum(knots[-1], raw_95_grid).astype(np.float64)
            les_100_after_floor_grid = knots[-1].astype(np.float64, copy=True)

            if percentile_99_idx is not None:
                violating_mask = knots[percentile_99_idx] > knots[-1]
                p99_violation_mask = violating_mask.astype(bool, copy=True)
                p99_les_minus_one_grid = (knots[-1] - 1.0).astype(np.float64, copy=True)
                if self.p99_violation_strategy == "les_minus_one":
                    replacement = p99_les_minus_one_grid
                    p99_choose_raw98_mask = np.zeros_like(violating_mask, dtype=bool)
                    p99_choose_les_minus_one_mask = violating_mask.astype(bool, copy=True)
                elif self.p99_violation_strategy == "raw98_or_les_minus_one":
                    replacement = np.minimum(raw_98_grid, p99_les_minus_one_grid)
                    p99_choose_raw98_mask = np.logical_and(violating_mask, raw_98_grid <= p99_les_minus_one_grid)
                    p99_choose_les_minus_one_mask = np.logical_and(violating_mask, np.logical_not(p99_choose_raw98_mask))
                else:
                    raise ValueError(f"Unsupported p99_violation_strategy: {self.p99_violation_strategy}")
                p99_replacement_grid = replacement.astype(np.float64, copy=True)
                p99_choice_grid = np.zeros(knots[-1].shape, dtype=np.int8)
                p99_choice_grid[p99_choose_raw98_mask] = 1
                p99_choice_grid[p99_choose_les_minus_one_mask] = 2
                knots[percentile_99_idx] = np.where(
                    violating_mask,
                    replacement,
                    knots[percentile_99_idx],
                ).astype(np.float64)
                p99_post_adjust_grid = knots[percentile_99_idx].astype(np.float64, copy=True)
            knots = enforce_knot_order(
                knots,
                min_interval=self.min_interval,
                mode=self.enforce_mode,
            )

        if self.use_message_passing:
            for _ in range(self.iteration):
                knots[0] = message_passing(knots[0], direction="decrease").astype(np.float64)
                knots[-1] = message_passing(knots[-1], direction="increase").astype(np.float64)
                # knots[1] = message_passing(knots[1], direction="decrease").astype(np.float64)
                # knots[1] = message_passing(knots[1], direction="increase").astype(np.float64)

                for idx in range(1, knots.shape[0] - 1):
                    knots[idx] = symmetric_message_passing(knots[idx],  alpha=self.mid_alpha)
                knots = enforce_knot_order(
                    knots,
                    min_interval=self.min_interval,
                    mode=self.enforce_mode,
                )
        else:
            knots = enforce_knot_order(
                knots,
                min_interval=self.min_interval,
                mode=self.enforce_mode,
            )

        return knots, {
            "frame_grid_p95_minus_p5_median": frame_grid_span_median,
            "frame_les_diff": frame_les_diff,
            "raw_95_grid": raw_95_grid,
            "raw_98_grid": raw_98_grid,
            "les_100_before_floor_grid": les_100_before_floor_grid,
            "les_100_after_floor_grid": les_100_after_floor_grid,
            "p99_pre_adjust_grid": p99_pre_adjust_grid,
            "p99_post_adjust_grid": p99_post_adjust_grid,
            "p99_violation_mask": p99_violation_mask,
            "p99_choice_grid": p99_choice_grid,
            "p99_replacement_grid": p99_replacement_grid,
            "p99_les_minus_one_grid": p99_les_minus_one_grid,
            "p99_choose_raw98_mask": p99_choose_raw98_mask,
            "p99_choose_les_minus_one_mask": p99_choose_les_minus_one_mask,
        }

    def process_debug(self, input: Union[str, np.ndarray]) -> dict:
        image = self._load_image(input)
        raw_knot_grids, knot_build_debug = self._build_knot_grids(image)
        knot_grids, temporal_debug = self._apply_knot_temporal_smoothing(raw_knot_grids)
        anchor_grids, anchor_debug = build_output_anchors(
            knot_stack=knot_grids,
            percentiles=self.percentiles,
            anchor_mode=self.anchor_mode,
            span_lambda=self.span_lambda,
            slope_cap_ratio=self.slope_cap_ratio,
            tail_split_percentile=self.tail_split_percentile,
            tail_suppression_gain=self.tail_suppression_gain,
            tail_activation_ratio=self.tail_activation_ratio,
            tail_min_output=self.tail_min_output,
            body_linear_cap_ratio=self.body_linear_cap_ratio,
            body_tail_slope_ratio_cap=self.body_tail_slope_ratio_cap,
        )

        knot_fields = resize_stack(knot_grids, image.shape)
        anchor_fields = resize_stack(anchor_grids, image.shape)

        rescaled_pre_post = piecewise_linear_rescale(
            image=image,
            knot_fields=knot_fields,
            anchor_fields=anchor_fields,
        )
        rescaled = self._postprocess_output(rescaled_pre_post)
        curve_overlay = build_grid_curve_overlay(
            image=rescaled,
            knot_grids=knot_grids,
            anchor_grids=anchor_grids,
            raw_image=image,
        )
        distribution_overlay = build_grid_distribution_overlay(
            image=rescaled,
            raw_image=image,
            grid_shape=self.grid_shape,
            knot_grids=knot_grids,
            percentiles=self.percentiles,
        )
        tail_ratio_overlay = build_grid_value_overlay(
            image=rescaled,
            value_grid=anchor_debug["tail_ratio_grid"],
            title="tail ratio",
            value_format="{:.3f}",
            active_mask=anchor_debug["tail_adjustment_active_mask"],
        )
        body_slope_gain_overlay = build_grid_value_overlay(
            image=rescaled,
            value_grid=anchor_debug["body_slope_gain_grid"],
            title="body slope x",
            value_format="{:.2f}x",
            active_mask=anchor_debug["tail_adjustment_active_mask"],
        )
        p99_choice_overlay = build_grid_choice_overlay(
            image=rescaled,
            choice_grid=knot_build_debug["p99_choice_grid"],
            title="p99 fix choice",
        )
        debug_board = build_grid_debug_board(
            image_shape=image.shape,
            anchor_mode=self.anchor_mode,
            tail_ratio_grid=anchor_debug["tail_ratio_grid"],
            body_span_grid=anchor_debug["body_span_grid"],
            tail_span_grid=anchor_debug["tail_span_grid"],
            body_output_width_grid=anchor_debug["body_output_width_grid"],
            tail_output_width_grid=anchor_debug["tail_output_width_grid"],
            tail_compression_grid=anchor_debug["tail_compression_grid"],
            body_slope_gain_grid=anchor_debug["body_slope_gain_grid"],
            tail_adjustment_active_mask=anchor_debug["tail_adjustment_active_mask"],
            linear_cap_mask=anchor_debug["linear_cap_mask"],
            p99_choice_grid=knot_build_debug["p99_choice_grid"],
            frame_grid_span_median=knot_build_debug["frame_grid_p95_minus_p5_median"],
            frame_les_diff=knot_build_debug["frame_les_diff"],
        )
        debug_preview = build_debug_preview(
            raw_image=_normalize_to_uint8(image),
            rescaled_image=rescaled,
            curve_overlay=curve_overlay,
            distribution_overlay=distribution_overlay,
            tail_ratio_overlay=tail_ratio_overlay,
            p99_choice_overlay=p99_choice_overlay,
            body_slope_gain_overlay=body_slope_gain_overlay,
        )
        combined_preview = build_side_by_side_preview(
            left_image=rescaled,
            right_image=curve_overlay,
        )

        return {
            "image": image,
            "percentiles": list(self.percentiles),
            "anchor_mode": self.anchor_mode,
            "tail_split_percentile": self.tail_split_percentile,
            "tail_suppression_gain": self.tail_suppression_gain,
            "tail_activation_ratio": self.tail_activation_ratio,
            "tail_min_output": self.tail_min_output,
            "body_linear_cap_ratio": self.body_linear_cap_ratio,
            "body_tail_slope_ratio_cap": self.body_tail_slope_ratio_cap,
            "use_les": self.use_les,
            "les_p95_floor": self.les_p95_floor,
            "p99_violation_strategy": self.p99_violation_strategy,
            "use_message_passing": self.use_message_passing,
            "global_percentiles": self.global_percentiles,
            "enforce_mode": self.enforce_mode,
            "raw_knot_grids": raw_knot_grids,
            "knot_grids": knot_grids,
            "anchor_grids": anchor_grids,
            "segment_widths": anchor_debug["segment_widths"],
            "segment_slopes": anchor_debug["segment_slopes"],
            "body_span_grid": anchor_debug["body_span_grid"],
            "tail_span_grid": anchor_debug["tail_span_grid"],
            "linear_body_width_grid": anchor_debug["linear_body_width_grid"],
            "linear_tail_width_grid": anchor_debug["linear_tail_width_grid"],
            "linear_body_slope_grid": anchor_debug["linear_body_slope_grid"],
            "linear_tail_slope_grid": anchor_debug["linear_tail_slope_grid"],
            "body_output_width_grid": anchor_debug["body_output_width_grid"],
            "tail_output_width_grid": anchor_debug["tail_output_width_grid"],
            "tail_ratio_grid": anchor_debug["tail_ratio_grid"],
            "tail_excess_grid": anchor_debug["tail_excess_grid"],
            "tail_compression_grid": anchor_debug["tail_compression_grid"],
            "body_slope_grid": anchor_debug["body_slope_grid"],
            "tail_slope_grid": anchor_debug["tail_slope_grid"],
            "body_slope_gain_grid": anchor_debug["body_slope_gain_grid"],
            "tail_slope_gain_grid": anchor_debug["tail_slope_gain_grid"],
            "body_tail_slope_ratio_grid": anchor_debug["body_tail_slope_ratio_grid"],
            "tail_adjustment_active_mask": anchor_debug["tail_adjustment_active_mask"],
            "linear_cap_mask": anchor_debug["linear_cap_mask"],
            "slope_ratio_cap_mask": anchor_debug["slope_ratio_cap_mask"],
            "knot_fields": knot_fields,
            "anchor_fields": anchor_fields,
            "rescaled_pre_post": rescaled_pre_post,
            "rescaled": rescaled,
            "curve_overlay": curve_overlay,
            "distribution_overlay": distribution_overlay,
            "tail_ratio_overlay": tail_ratio_overlay,
            "body_slope_gain_overlay": body_slope_gain_overlay,
            "p99_choice_overlay": p99_choice_overlay,
            "debug_board": debug_board,
            "debug_preview": debug_preview,
            "combined_preview": combined_preview,
            **knot_build_debug,
            **temporal_debug,
        }

    def __call__(self, input: Union[str, np.ndarray]) -> np.ndarray:
        return self.process_debug(input)["rescaled"]


if __name__ == "__main__":
    raise SystemExit(
        "Import MultiKnotPercentileFieldscale and call it with a thermal ndarray or image path."
    )
