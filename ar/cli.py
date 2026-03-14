"""CLI entry point for AutoResearch."""

import argparse
import os
import sys


def cmd_detect(args):
    from .platform import detect_platform
    platform = detect_platform(device_override=args.device)
    print(platform.summary())


def cmd_init(args):
    from .platform import detect_platform
    from .config import TaskConfig
    from .program_gen import generate_program
    from .templates import list_templates, scaffold_template

    dest = os.getcwd()

    if args.task:
        templates = list_templates()
        if args.task not in templates:
            available = ", ".join(templates.keys())
            print(f"Unknown template '{args.task}'. Available: {available}")
            sys.exit(1)

        print(f"Scaffolding template: {args.task}")
        print()
        scaffold_template(args.task, dest)
        print()

    if args.source and os.path.exists(args.source):
        from .config import TaskConfig
        config = TaskConfig.from_yaml(args.source)
    elif os.path.exists(os.path.join(dest, "task.yaml")):
        from .config import TaskConfig
        config = TaskConfig.from_yaml(os.path.join(dest, "task.yaml"))
    else:
        print("No task.yaml found. Use --task <template> or --from <task.yaml>")
        sys.exit(1)

    platform = detect_platform(device_override=config.device)

    # Auto-fill time budget if not set
    if not config.time_limit_seconds:
        defaults = platform.compute_defaults(config.model_class)
        config.time_limit_seconds = defaults.get("time_budget", 120)
        config.to_yaml(os.path.join(dest, "task.yaml"))

    # Generate program.md
    program = generate_program(config, platform)
    program_path = os.path.join(dest, "program.md")
    if not os.path.exists(program_path) or args.task:
        with open(program_path, "w") as f:
            f.write(program)
        print(f"  Created program.md (platform-aware agent instructions)")

    print()
    print(f"Project ready: {config.name}")
    print(f"Platform: {platform.device_name} ({platform.device})")
    print(f"Metric: {config.metric_name} ({config.metric_direction})")
    print(f"Time budget: {config.time_limit_seconds}s per experiment")
    print()
    print("Next steps:")
    print(f"  1. Review task.yaml")
    print(f"  2. Run '{config.eval_command}' to verify training works")
    print(f"  3. Run 'ar run' to start autonomous research")
    print(f"  4. Or point your coding agent to program.md")


def cmd_status(args):
    from .tracker import ExperimentTracker

    results_path = args.results or "results.jsonl"
    tracker = ExperimentTracker(results_path)
    print(tracker.summary())


def cmd_analyze(args):
    from .analysis import generate_report

    results_path = args.results or "results.jsonl"
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run some experiments first.")
        sys.exit(1)
    output_dir = args.output or "."
    generate_report(results_path, output_dir)


def cmd_run(args):
    from .platform import detect_platform
    from .config import TaskConfig
    from .runner import ExperimentRunner
    from .tracker import ExperimentTracker, ExperimentRecord
    from .metrics import MetricResult
    import time as time_mod

    config_path = "task.yaml"
    if not os.path.exists(config_path):
        print("No task.yaml found. Run 'ar init' first.")
        sys.exit(1)

    config = TaskConfig.from_yaml(config_path)
    platform = detect_platform(device_override=config.device)
    runner = ExperimentRunner(config)
    tracker = ExperimentTracker("results.jsonl")

    max_exp = args.max_experiments or config.max_experiments or 1

    print(f"AutoResearch — {config.name} on {platform.device_name} ({platform.device})")
    print(f"Metric: {config.metric_name} ({config.metric_direction})")
    print(f"Time budget: {config.time_limit_seconds}s per experiment")
    print(f"Running up to {max_exp} experiment(s)")
    print()

    for i in range(max_exp):
        exp_id = tracker.next_id
        is_baseline = tracker.count == 0
        label = "Baseline" if is_baseline else f"Experiment {exp_id}"

        print(f"[exp {exp_id}] {label}")
        print(f"  Running {config.eval_command}...", end="", flush=True)

        result = runner.run()

        if not result.success:
            print(f" CRASHED (exit code {result.returncode})")
            if result.error_tail:
                # Show last few lines of error
                lines = result.error_tail.strip().split("\n")
                for line in lines[-5:]:
                    print(f"    {line}")
            rec = ExperimentRecord(
                experiment_id=exp_id,
                timestamp=time_mod.strftime("%Y-%m-%dT%H:%M:%SZ"),
                commit="",
                hypothesis="baseline" if is_baseline else "",
                category="baseline" if is_baseline else "",
                change_summary="baseline" if is_baseline else "crash",
                files_changed=[],
                metric_name=config.metric_name,
                metric_value=0.0,
                metric_direction=config.metric_direction,
                secondary_metrics={},
                status="crash",
                platform_device=platform.device,
                platform_name=platform.device_name,
            )
            tracker.record(rec)
            continue

        metric = result.metric
        mem = result.peak_memory_mb
        print(f" done ({result.elapsed:.0f}s)")
        print(f"  {metric.name}: {metric.value:.6f}  peak_memory: {mem:.0f} MB", end="")

        # Determine keep/discard
        best = tracker.best_result
        if best is None:
            status = "keep"
            print(f"  status: keep (baseline)")
        elif metric.is_better_than(MetricResult(best.metric_name, best.metric_value, best.metric_direction)):
            delta = metric.value - best.metric_value
            status = "keep"
            print(f"  status: keep ({delta:+.6f})")
        else:
            status = "discard"
            print(f"  status: discard")

        rec = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=time_mod.strftime("%Y-%m-%dT%H:%M:%SZ"),
            commit="",
            hypothesis="baseline" if is_baseline else "",
            category="baseline" if is_baseline else "",
            change_summary="baseline" if is_baseline else "",
            files_changed=[],
            metric_name=config.metric_name,
            metric_value=metric.value,
            metric_direction=config.metric_direction,
            secondary_metrics=metric.secondary,
            status=status,
            platform_device=platform.device,
            platform_name=platform.device_name,
        )
        tracker.record(rec)

    print()
    print(tracker.summary())


def main():
    parser = argparse.ArgumentParser(
        prog="ar",
        description="AutoResearch — Platform-agnostic autonomous ML research",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ar detect
    p_detect = subparsers.add_parser("detect", help="Show detected platform and recommendations")
    p_detect.add_argument("--device", default=None, help="Force device (cpu, mps, cuda)")

    # ar init
    p_init = subparsers.add_parser("init", help="Scaffold a new research project")
    p_init.add_argument("--task", default=None, help="Template name (tabular, tiny-lm, ...)")
    p_init.add_argument("--from", dest="source", default=None, help="Path to existing task.yaml")

    # ar status
    p_status = subparsers.add_parser("status", help="Show experiment results summary")
    p_status.add_argument("--results", default=None, help="Path to results.jsonl")

    # ar run
    p_run = subparsers.add_parser("run", help="Run experiments")
    p_run.add_argument("--max-experiments", type=int, default=0, help="Maximum experiments to run (0=use config)")

    # ar analyze
    p_analyze = subparsers.add_parser("analyze", help="Generate analysis report and progress chart")
    p_analyze.add_argument("--results", default=None, help="Path to results.jsonl")
    p_analyze.add_argument("--output", default=None, help="Output directory for charts")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "detect": cmd_detect,
        "init": cmd_init,
        "status": cmd_status,
        "run": cmd_run,
        "analyze": cmd_analyze,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
