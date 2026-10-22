from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import os

import cv2

if __package__:
    from .fieldscale_multiknot import MultiKnotPercentileFieldscale, parse_percentiles
else:
    from fieldscale_multiknot import MultiKnotPercentileFieldscale, parse_percentiles


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class BatchConfig:
    output_dir: str
    overlay_dir: str | None
    combined_dir: str | None
    debug_dir: str | None
    max_diff: float
    min_diff: float
    iteration: int
    gamma: float
    clahe: bool
    video: bool
    ema_period: int
    ema_alpha_override: float | None
    les_mode: str
    grid_shape: tuple[int, int]
    percentiles: tuple[float, ...]
    mid_alpha: float
    min_interval: float
    anchor_mode: str
    span_lambda: float
    slope_cap_ratio: float | None
    tail_split_percentile: float
    tail_suppression_gain: float
    tail_activation_ratio: float
    tail_min_output: float
    body_linear_cap_ratio: float | None
    body_tail_slope_ratio_cap: float | None
    use_les: bool
    les_p95_floor: bool
    p99_violation_strategy: str
    use_message_passing: bool
    global_percentiles: bool
    enforce_mode: str


def parse_pair(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Expected rows,cols format.")
    values = (int(parts[0]), int(parts[1]))
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("Grid values must be positive.")
    return values


def parse_percentile_text(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parse_percentiles(float(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescale thermal frames with the FIELDSCALE 2.0 three-knot paper preset."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of raw grayscale thermal frames.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate folder for 8-bit frames and run metadata.")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--combined-dir", type=Path, default=None)
    parser.add_argument("--save-debug-overlays", action="store_true")
    parser.add_argument("--max-diff", type=float, default=500.0)
    parser.add_argument("--min-diff", type=float, default=500.0)
    parser.add_argument("--iteration", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--clahe", action="store_true", default=True)
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--ema-period", type=int, default=10)
    parser.add_argument("--ema-alpha-override", type=float, default=None)
    parser.add_argument("--les-mode", choices=["single", "repeat5"], default="single")
    parser.add_argument("--grid-shape", type=str, default="8,8")
    parser.add_argument("--percentiles", type=str, default="0,99,100")
    parser.add_argument("--mid-alpha", type=float, default=0.5)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--anchor-mode", choices=["span", "tail_adaptive_split"], default="tail_adaptive_split")
    parser.add_argument("--span-lambda", type=float, default=1.0)
    parser.add_argument("--slope-cap-ratio", type=float, default=4.0)
    parser.add_argument("--tail-split-percentile", type=float, default=99.0)
    parser.add_argument("--tail-suppression-gain", type=float, default=1.0)
    parser.add_argument("--tail-activation-ratio", type=float, default=0.3)
    parser.add_argument("--tail-min-output", type=float, default=4.0)
    parser.add_argument("--body-linear-cap-ratio", type=float, default=1.5)
    parser.add_argument("--body-tail-slope-ratio-cap", type=float, default=2.0)
    les_group = parser.add_mutually_exclusive_group()
    les_group.add_argument("--use-les", dest="use_les", action="store_true")
    les_group.add_argument("--no-les", dest="use_les", action="store_false")
    parser.add_argument("--no-les-p95-floor", action="store_true")
    parser.add_argument(
        "--p99-violation-strategy",
        choices=["raw98_or_les_minus_one", "les_minus_one"],
        default="raw98_or_les_minus_one",
    )
    mp_group = parser.add_mutually_exclusive_group()
    mp_group.add_argument("--use-mp", dest="use_mp", action="store_true")
    mp_group.add_argument("--no-mp", dest="use_mp", action="store_false")
    parser.set_defaults(use_les=True, use_mp=True)
    parser.add_argument("--global-percentiles", action="store_true")
    parser.add_argument("--enforce-mode", choices=["bottom_up", "top_down"], default="bottom_up")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory")
    input_root = args.input_dir.resolve()
    for name in ("output_dir", "overlay_dir", "combined_dir"):
        destination = getattr(args, name)
        if destination is not None and destination.resolve() == input_root:
            parser.error(f"--{name.replace('_', '-')} must differ from --input-dir")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.ema_period < 1:
        parser.error("--ema-period must be at least 1")
    if args.ema_alpha_override is not None and not 0 < args.ema_alpha_override <= 1:
        parser.error("--ema-alpha-override must be in (0, 1]")
    return args


def iter_frame_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    frame_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if limit is not None:
        return frame_files[:max(0, int(limit))]
    return frame_files


def build_processor(config: BatchConfig) -> MultiKnotPercentileFieldscale:
    return MultiKnotPercentileFieldscale(
        max_diff=config.max_diff,
        min_diff=config.min_diff,
        iteration=config.iteration,
        gamma=config.gamma,
        clahe=config.clahe,
        video=config.video,
        ema_period=config.ema_period,
        ema_alpha_override=config.ema_alpha_override,
        les_mode=config.les_mode,
        grid_shape=config.grid_shape,
        percentiles=config.percentiles,
        mid_alpha=config.mid_alpha,
        min_interval=config.min_interval,
        anchor_mode=config.anchor_mode,
        span_lambda=config.span_lambda,
        slope_cap_ratio=config.slope_cap_ratio,
        tail_split_percentile=config.tail_split_percentile,
        tail_suppression_gain=config.tail_suppression_gain,
        tail_activation_ratio=config.tail_activation_ratio,
        tail_min_output=config.tail_min_output,
        body_linear_cap_ratio=config.body_linear_cap_ratio,
        body_tail_slope_ratio_cap=config.body_tail_slope_ratio_cap,
        use_les=config.use_les,
        les_p95_floor=config.les_p95_floor,
        p99_violation_strategy=config.p99_violation_strategy,
        use_message_passing=config.use_message_passing,
        global_percentiles=config.global_percentiles,
        enforce_mode=config.enforce_mode,
    )


def process_frame(
    frame_path: Path,
    config: BatchConfig,
    processor: MultiKnotPercentileFieldscale,
) -> dict[str, object]:
    debug = processor.process_debug(str(frame_path))
    output = debug["rescaled"]
    output_path = Path(config.output_dir) / frame_path.name
    ok = cv2.imwrite(str(output_path), output)
    if not ok:
        raise IOError(f"Failed to write output image: {output_path}")

    overlay_path = None
    if config.overlay_dir is not None:
        overlay_path = Path(config.overlay_dir) / frame_path.name
        ok = cv2.imwrite(str(overlay_path), debug["curve_overlay"])
        if not ok:
            raise IOError(f"Failed to write overlay image: {overlay_path}")

    combined_path = None
    if config.combined_dir is not None:
        combined_path = Path(config.combined_dir) / frame_path.name
        ok = cv2.imwrite(str(combined_path), debug["combined_preview"])
        if not ok:
            raise IOError(f"Failed to write combined image: {combined_path}")

    curve_debug_path = None
    distribution_debug_path = None
    tail_ratio_debug_path = None
    body_slope_gain_debug_path = None
    p99_choice_debug_path = None
    board_debug_path = None
    preview_debug_path = None
    if config.debug_dir is not None:
        debug_dir = Path(config.debug_dir)
        curve_debug_path = debug_dir / f"{frame_path.stem}__curve.png"
        distribution_debug_path = debug_dir / f"{frame_path.stem}__distribution.png"
        tail_ratio_debug_path = debug_dir / f"{frame_path.stem}__tail_ratio.png"
        body_slope_gain_debug_path = debug_dir / f"{frame_path.stem}__body_slope_gain.png"
        p99_choice_debug_path = debug_dir / f"{frame_path.stem}__p99_choice.png"
        board_debug_path = debug_dir / f"{frame_path.stem}__board.png"
        preview_debug_path = debug_dir / f"{frame_path.stem}__debug.png"

        debug_outputs = [
            (curve_debug_path, debug["curve_overlay"]),
            (distribution_debug_path, debug["distribution_overlay"]),
            (tail_ratio_debug_path, debug["tail_ratio_overlay"]),
            (body_slope_gain_debug_path, debug["body_slope_gain_overlay"]),
            (p99_choice_debug_path, debug["p99_choice_overlay"]),
            (board_debug_path, debug["debug_board"]),
            (preview_debug_path, debug["debug_preview"]),
        ]
        for save_path, image_data in debug_outputs:
            ok = cv2.imwrite(str(save_path), image_data)
            if not ok:
                raise IOError(f"Failed to write debug image: {save_path}")

    return {
        "frame_name": frame_path.name,
        "frame_path": str(frame_path),
        "output_path": str(output_path),
        "overlay_path": None if overlay_path is None else str(overlay_path),
        "combined_path": None if combined_path is None else str(combined_path),
        "curve_debug_path": None if curve_debug_path is None else str(curve_debug_path),
        "distribution_debug_path": None if distribution_debug_path is None else str(distribution_debug_path),
        "tail_ratio_debug_path": None if tail_ratio_debug_path is None else str(tail_ratio_debug_path),
        "body_slope_gain_debug_path": None if body_slope_gain_debug_path is None else str(body_slope_gain_debug_path),
        "p99_choice_debug_path": None if p99_choice_debug_path is None else str(p99_choice_debug_path),
        "board_debug_path": None if board_debug_path is None else str(board_debug_path),
        "preview_debug_path": None if preview_debug_path is None else str(preview_debug_path),
        "use_message_passing": bool(config.use_message_passing),
        "output_mean": float(output.mean()),
        "output_min": int(output.min()),
        "output_max": int(output.max()),
    }


def analyze_one(frame_path: Path, config: BatchConfig) -> dict[str, object]:
    processor = build_processor(config)
    return process_frame(frame_path, config, processor)


def infer_sequence_key(frame_path: Path) -> str:
    stem = frame_path.stem
    if "-frame-" in stem:
        return stem.split("-frame-", maxsplit=1)[0]
    if "_frame" in stem:
        return stem.split("_frame", maxsplit=1)[0]
    return "__single_sequence__"


def save_manifest(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, object]], args: argparse.Namespace, config: BatchConfig) -> dict[str, object]:
    total_frames = len(rows)
    return {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "total_frames": total_frames,
        "params": {
            **asdict(config),
            "grid_shape": list(config.grid_shape),
            "percentiles": list(config.percentiles),
        },
        "mean_output_mean": sum(float(row["output_mean"]) for row in rows) / total_frames,
    }


def main() -> None:
    args = parse_args()
    frame_files = iter_frame_files(args.input_dir, args.limit)
    if not frame_files:
        raise FileNotFoundError(f"No supported image files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = None if args.overlay_dir is None else args.overlay_dir
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = None if args.combined_dir is None else args.combined_dir
    if combined_dir is not None:
        combined_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug_overlays" if args.save_debug_overlays else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    config = BatchConfig(
        output_dir=str(args.output_dir),
        overlay_dir=None if overlay_dir is None else str(overlay_dir),
        combined_dir=None if combined_dir is None else str(combined_dir),
        debug_dir=None if debug_dir is None else str(debug_dir),
        max_diff=float(args.max_diff),
        min_diff=float(args.min_diff),
        iteration=int(args.iteration),
        gamma=float(args.gamma),
        clahe=bool(args.clahe and not args.no_clahe),
        video=bool(args.video),
        ema_period=int(args.ema_period),
        ema_alpha_override=None if args.ema_alpha_override is None else float(args.ema_alpha_override),
        les_mode=str(args.les_mode),
        grid_shape=parse_pair(args.grid_shape),
        percentiles=parse_percentile_text(args.percentiles),
        mid_alpha=float(args.mid_alpha),
        min_interval=float(args.min_interval),
        anchor_mode=str(args.anchor_mode),
        span_lambda=float(args.span_lambda),
        slope_cap_ratio=float(args.slope_cap_ratio),
        tail_split_percentile=float(args.tail_split_percentile),
        tail_suppression_gain=float(args.tail_suppression_gain),
        tail_activation_ratio=float(args.tail_activation_ratio),
        tail_min_output=float(args.tail_min_output),
        body_linear_cap_ratio=float(args.body_linear_cap_ratio),
        body_tail_slope_ratio_cap=None if args.body_tail_slope_ratio_cap is None else float(args.body_tail_slope_ratio_cap),
        use_les=bool(args.use_les),
        les_p95_floor=not bool(args.no_les_p95_floor),
        p99_violation_strategy=str(args.p99_violation_strategy),
        use_message_passing=bool(args.use_mp),
        global_percentiles=bool(args.global_percentiles),
        enforce_mode=str(args.enforce_mode),
    )

    if config.video and args.workers > 1:
        print("video=True detected; forcing sequential processing to preserve EMA state.", flush=True)

    if args.workers <= 1 or config.video:
        rows = []
        processor = build_processor(config)
        current_sequence_key = None
        for index, frame_path in enumerate(frame_files, start=1):
            sequence_key = infer_sequence_key(frame_path)
            if sequence_key != current_sequence_key:
                processor.reset_temporal_state()
                current_sequence_key = sequence_key
            row = process_frame(frame_path, config, processor)
            rows.append(row)
            if index == 1 or index % 100 == 0 or index == len(frame_files):
                print(f"[{index}/{len(frame_files)}] saved {Path(row['output_path']).name}", flush=True)
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for index, row in enumerate(executor.map(analyze_one, frame_files, [config] * len(frame_files)), start=1):
                rows.append(row)
                if index == 1 or index % 100 == 0 or index == len(frame_files):
                    print(f"[{index}/{len(frame_files)}] saved {Path(row['output_path']).name}", flush=True)

    manifest_path = args.output_dir / "manifest.csv"
    summary_path = args.output_dir / "summary.json"
    save_manifest(rows, manifest_path)
    summary = build_summary(rows, args, config)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "total_frames": len(rows),
        "params": summary["params"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
