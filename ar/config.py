"""Task configuration from task.yaml."""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class TaskConfig:
    # Task metadata
    name: str = "unnamed"
    description: str = ""

    # Dataset
    dataset_source: str = ""
    dataset_prepare_script: str = "prepare.py"
    dataset_cache_dir: str = ""

    # Model
    train_script: str = "train.py"
    editable_files: list = field(default_factory=lambda: ["train.py"])
    readonly_files: list = field(default_factory=lambda: ["prepare.py"])
    model_class: str = "custom"

    # Evaluation
    metric_name: str = "val_loss"
    metric_direction: str = "minimize"
    eval_command: str = "uv run train.py"
    extract_patterns: dict = field(default_factory=dict)

    # Budget
    time_limit_seconds: int = 0  # 0 = auto (platform-aware)
    max_experiments: int = 0     # 0 = unlimited

    # Platform
    device: str = "auto"

    # Constraints
    no_new_dependencies: bool = True
    no_modify_eval: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "TaskConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()

        if task := raw.get("task"):
            config.name = task.get("name", config.name)
            config.description = task.get("description", config.description)

        if ds := raw.get("dataset"):
            config.dataset_source = ds.get("source", config.dataset_source)
            config.dataset_prepare_script = ds.get("prepare_script", config.dataset_prepare_script)
            config.dataset_cache_dir = ds.get("cache_dir", config.dataset_cache_dir)

        if model := raw.get("model"):
            config.train_script = model.get("train_script", config.train_script)
            config.editable_files = model.get("editable_files", config.editable_files)
            config.readonly_files = model.get("readonly_files", config.readonly_files)
            config.model_class = model.get("model_class", config.model_class)

        if ev := raw.get("evaluation"):
            config.metric_name = ev.get("metric", config.metric_name)
            config.metric_direction = ev.get("direction", config.metric_direction)
            config.eval_command = ev.get("eval_command", config.eval_command)
            if pat := ev.get("extract_pattern"):
                config.extract_patterns[config.metric_name] = pat

        if budget := raw.get("budget"):
            tl = budget.get("time_limit_seconds", 0)
            config.time_limit_seconds = 0 if tl == "auto" else int(tl)
            me = budget.get("max_experiments", 0)
            config.max_experiments = 0 if me in ("unlimited", "auto") else int(me)

        if plat := raw.get("platform"):
            config.device = plat.get("device", config.device)

        if constraints := raw.get("constraints"):
            config.no_new_dependencies = constraints.get("no_new_dependencies", True)
            config.no_modify_eval = constraints.get("no_modify_eval", True)

        return config

    def to_yaml(self, path: str):
        data = {
            "task": {"name": self.name, "description": self.description},
            "dataset": {
                "source": self.dataset_source,
                "prepare_script": self.dataset_prepare_script,
            },
            "model": {
                "train_script": self.train_script,
                "editable_files": self.editable_files,
                "readonly_files": self.readonly_files,
                "model_class": self.model_class,
            },
            "evaluation": {
                "metric": self.metric_name,
                "direction": self.metric_direction,
                "eval_command": self.eval_command,
            },
            "budget": {
                "time_limit_seconds": self.time_limit_seconds or "auto",
                "max_experiments": self.max_experiments or "unlimited",
            },
            "platform": {"device": self.device},
            "constraints": {
                "no_new_dependencies": self.no_new_dependencies,
                "no_modify_eval": self.no_modify_eval,
            },
        }
        if self.dataset_cache_dir:
            data["dataset"]["cache_dir"] = self.dataset_cache_dir
        if self.extract_patterns:
            data["evaluation"]["extract_patterns"] = self.extract_patterns

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
