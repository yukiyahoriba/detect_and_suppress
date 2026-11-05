"""Durable episode commits, strict resume checks, and shard aggregation."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shutil

from .config import EvaluationConfig, Shard, canonical


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def summarize(rows: list[dict], expected: int) -> dict:
    if len({r["rollout_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate rollout results")
    successes = sum(r["success"] for r in rows)
    by_task = []
    for task_id in range(10):
        task = [r for r in rows if r["task_id"] == task_id]
        n = sum(r["success"] for r in task)
        by_task.append(dict(task_id=task_id, task=task[0]["task"] if task else None,
                            successes=n, episodes=len(task), success_rate=n / len(task) if task else 0.))
    counts = {key: sum(r.get(key, 0) for r in rows) for key in (
        "num_forwards", "num_detected_forwards", "num_patch_forwards", "num_patch_detected_forwards",
        "num_clean_forwards", "num_clean_detected_forwards", "num_intervention_forwards")}
    return dict(successes=successes, episodes_completed=len(rows), episodes_expected=expected,
                success_rate=successes / len(rows) if rows else 0., complete=len(rows) == expected,
                by_task=by_task, detector={**counts,
                    "tpr": counts["num_patch_detected_forwards"] / counts["num_patch_forwards"] if counts["num_patch_forwards"] else 0.,
                    "fpr": counts["num_clean_detected_forwards"] / counts["num_clean_forwards"] if counts["num_clean_forwards"] else 0.})


class ResultStore:
    """One writer per directory; only committed, validated episodes are skipped."""

    def __init__(self, directory: str | Path, config: EvaluationConfig, shard: Shard = Shard()):
        self.directory, self.config, self.shard = Path(directory), config, shard
        self.rows: list[dict] = []
        self._lock = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / ".writer.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name, value in (("config.json", self.config.to_dict()), ("partition.json", asdict(self.shard))):
                path = self.directory / name
                if path.exists():
                    previous = json.loads(path.read_text())
                    if name == "config.json":
                        previous = EvaluationConfig.from_dict(previous).to_dict()
                    if canonical(previous) != canonical(value):
                        raise ValueError(f"configuration mismatch: {path}")
                    if name == "config.json" and "protocol" not in json.loads(path.read_text()):
                        atomic_json(path, value)
                elif (self.directory / "results.jsonl").exists():
                    raise ValueError("cannot resume results without their configuration")
                else:
                    atomic_json(path, value)
            path = self.directory / "results.jsonl"
            self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            for row in self.rows:
                self._validate(row)
                events_path = self.directory / row["events_file"]
                events = json.loads(events_path.read_text())
                if events["config_id"] != self.config.fingerprint or events["rollout_id"] != row["rollout_id"]:
                    raise ValueError("event identity mismatch")
                self._validate_events(row, events["forwards"])
            if len(self.completed) != len(self.rows):
                raise ValueError("duplicate rollout ids; refusing unsafe resume")
            self.write_summary()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @property
    def completed(self) -> set[int]:
        return {r["rollout_id"] for r in self.rows}

    @property
    def summary(self) -> dict:
        return {"config_id": self.config.fingerprint, **summarize(self.rows, len(self.shard.rollout_ids))}

    def _validate(self, row):
        rid = row["rollout_id"]
        if type(rid) is not int or rid not in self.shard.rollout_ids or row["config_id"] != self.config.fingerprint:
            raise ValueError("result identity/shard mismatch")
        if row["task_id"] != rid // 50 or row["init_state_index"] != rid % 50 or type(row["success"]) is not bool:
            raise ValueError("invalid episode identity/success")
        if row["events_file"] != f"events/{rid:03d}.json":
            raise ValueError("invalid event path")
        if not 0 < row["num_steps"] <= self.config.max_steps:
            raise ValueError("invalid episode length")
        if self.config.required_patch_forwards is not None:
            if row["num_patch_forwards"] != self.config.required_patch_forwards:
                raise ValueError("SmolVLA exact patch-forward invariant failed")

    def commit(self, row: dict, events: list[dict]) -> None:
        if self._lock is None:
            raise RuntimeError("use ResultStore as a context manager")
        row = {**row, "config_id": self.config.fingerprint,
               "events_file": f"events/{row['rollout_id']:03d}.json"}
        self._validate(row)
        if row["rollout_id"] in self.completed:
            raise ValueError("episode already committed")
        self._validate_events(row, events)
        (self.directory / "events").mkdir(exist_ok=True)
        atomic_json(self.directory / row["events_file"], dict(config_id=self.config.fingerprint,
                    rollout_id=row["rollout_id"], forwards=events))
        # Events are durable before the episode completion marker is appended.
        with (self.directory / "results.jsonl").open("a") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)
        self.write_summary()

    def _validate_events(self, row, events):
        if len(events) != row["num_forwards"] or not events:
            raise ValueError("forward event count mismatch")
        previous_end = 0
        for event in events:
            if (event["forward_step"] != previous_end or event["action_start"] != previous_end or
                    not previous_end < event["action_end_exclusive"] <= row["num_steps"]):
                raise ValueError("invalid forward/action boundaries")
            if not math.isfinite(event["score"]):
                raise ValueError("invalid detector score")
            previous_end = event["action_end_exclusive"]
        if previous_end != row["num_steps"]:
            raise ValueError("missing final action boundary")
        checks = {"num_detected_forwards": sum(e["detected"] for e in events),
                  "num_patch_forwards": sum(e["patch_active"] for e in events),
                  "num_patch_detected_forwards": sum(e["patch_active"] and e["detected"] for e in events),
                  "num_clean_forwards": sum(not e["patch_active"] for e in events),
                  "num_clean_detected_forwards": sum(not e["patch_active"] and e["detected"] for e in events),
                  "num_intervention_forwards": sum(e["applied"] for e in events),
                  "intervention_calls": sum(e["intervention_calls"] for e in events)}
        if any(row[key] != value for key, value in checks.items()):
            raise ValueError("event/result counts differ")

    def write_summary(self):
        atomic_json(self.directory / "summary.json", self.summary)

    def __exit__(self, *_):
        if self._lock is not None:
            self._lock.close()
            self._lock = None


def merge_shards(config: EvaluationConfig, sources: list[str | Path], output: str | Path,
                 *, require_complete: bool = True) -> dict:
    """Merge validated disjoint shard results with the same experiment identity."""
    rows, events, seen = [], {}, set()
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source:
            raise ValueError("merge output must differ from input shards")
        shard = Shard(**json.loads((source / "partition.json").read_text()))
        with ResultStore(source, config, shard) as store:
            for row in store.rows:
                rid = row["rollout_id"]
                if rid in seen:
                    raise ValueError("overlapping shard results")
                seen.add(rid)
                rows.append(row)
                events[rid] = json.loads((source / row["events_file"]).read_text())["forwards"]
    if require_complete and seen != set(range(500)):
        raise ValueError(f"incomplete evaluation: {len(seen)}/500")
    with ResultStore(output, config) as merged:
        if merged.rows:
            raise ValueError("merge destination already contains results")
        for row in sorted(rows, key=lambda r: r["rollout_id"]):
            merged.commit(row, events[row["rollout_id"]])
        return merged.summary


def repartition_shards(config: EvaluationConfig, sources: list[str | Path], output: str | Path,
                       count: int) -> list[Path]:
    """Reassign committed episodes to new modulo shards after workers have stopped.

    Inputs remain unchanged. Output must be new; partially copied output is
    removed on failure. Active input writers are rejected by ResultStore.
    """
    Shard(0, count)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows, events, seen = [], {}, set()
    with ExitStack() as locks:
        for source in sources:
            source = Path(source).resolve()
            if source in output.parents:
                raise ValueError("repartition output must be outside input shards")
            partition = Shard(**json.loads((source / "partition.json").read_text()))
            store = locks.enter_context(ResultStore(source, config, partition))
            for row in store.rows:
                rid = row["rollout_id"]
                if rid in seen:
                    raise ValueError("overlapping input results")
                seen.add(rid)
                rows.append(row)
                events[rid] = json.loads((source / row["events_file"]).read_text())["forwards"]
        output.mkdir(parents=True)
        try:
            paths = [output / f"shard{i}" for i in range(count)]
            for index, path in enumerate(paths):
                with ResultStore(path, config, Shard(index, count)) as store:
                    for row in sorted(rows, key=lambda r: r["rollout_id"]):
                        if row["rollout_id"] % count == index:
                            store.commit(row, events[row["rollout_id"]])
            return paths
        except BaseException:
            shutil.rmtree(output)
            raise
