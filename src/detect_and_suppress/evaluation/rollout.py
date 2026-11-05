"""LIBERO episode execution with model-specific inference and environment adapters."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Callable

import numpy as np

from .config import EvaluationConfig, Shard
from .patches import overlay_active_timesteps, should_overlay
from .results import ResultStore


@dataclass
class Prediction:
    actions: np.ndarray
    score: float
    detected: bool
    applied: bool
    intervention_calls: int
    normalized_actions: object | None = None


class LiberoAdapter(Protocol):
    config: EvaluationConfig
    task_language: str
    flush_on_patch_change: bool

    def open_task(self, task_id: int, first_init: int) -> None: ...
    def reset_episode(self, rollout_id: int): ...
    def prepare_observation(self, observation, patch_active: bool, rollout_id: int, timestep: int): ...
    def predict(self, observation, patch_active: bool) -> Prediction: ...
    def step(self, action: np.ndarray) -> tuple[object, bool, bool]: ...
    def close_task(self) -> None: ...


def run_episode(adapter: LiberoAdapter, rollout_id: int, *, trace: Callable | None = None) -> tuple[dict, list[dict]]:
    """Execute one initialized task; trace receives images/chunks/actions for comparison."""
    config = adapter.config
    observation = adapter.reset_episode(rollout_id)
    active_steps = overlay_active_timesteps(config.schedule_horizon, config.overlay(), rollout_id)
    plan = deque()
    previous = None
    events = []
    active_event = None
    executed_patch_steps, intervened_steps = [], []
    success = ended = False
    for timestep in range(config.max_steps):
        patch_active = should_overlay(timestep, active_steps)
        # Pi0.5 invalidates queued actions on either visibility transition.
        # SmolVLA's paper condition replans every environment step.
        if adapter.flush_on_patch_change and previous is not None and previous != patch_active:
            plan.clear()
        previous = patch_active
        if patch_active:
            executed_patch_steps.append(timestep)
        prepared = adapter.prepare_observation(observation, patch_active, rollout_id, timestep)
        if trace:
            trace("observation", timestep, dict(rollout_id=rollout_id, observation=prepared, patch_active=patch_active))
        if not plan:
            prediction = adapter.predict(prepared, patch_active)
            chunk = np.asarray(prediction.actions, dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                raise RuntimeError("invalid policy action chunk")
            plan.extend(chunk[:config.replan_steps])
            active_event = dict(forward_step=timestep, patch_active=bool(patch_active), score=prediction.score,
                                detected=prediction.detected, applied=prediction.applied,
                                intervention_calls=prediction.intervention_calls, action_start=timestep,
                                action_end_exclusive=timestep)
            events.append(active_event)
            if trace:
                trace("forward", timestep, dict(rollout_id=rollout_id, observation=prepared, prediction=prediction))
        action = np.asarray(plan.popleft(), dtype=np.float32)
        if active_event["applied"]:
            intervened_steps.append(timestep)
        if trace:
            trace("step", timestep, dict(rollout_id=rollout_id, action=action.copy(), patch_active=patch_active))
        observation, ended, success = adapter.step(action)
        active_event["action_end_exclusive"] = timestep + 1
        if ended:
            break
    patch = [e for e in events if e["patch_active"]]
    clean = [e for e in events if not e["patch_active"]]
    if config.required_patch_forwards is not None:
        if len(patch) != config.required_patch_forwards:
            raise RuntimeError("episode ended before the required SmolVLA patch-forward count")
    row = dict(rollout_id=rollout_id, task_id=rollout_id // 50, init_state_index=rollout_id % 50,
               task=adapter.task_language, success=bool(success), num_steps=timestep + 1,
               ended=bool(ended), num_forwards=len(events), num_detected_forwards=sum(e["detected"] for e in events),
               num_patch_forwards=len(patch), num_patch_detected_forwards=sum(e["detected"] for e in patch),
               num_clean_forwards=len(clean), num_clean_detected_forwards=sum(e["detected"] for e in clean),
               num_intervention_forwards=sum(e["applied"] for e in events),
               intervention_calls=sum(e["intervention_calls"] for e in events),
               patch_steps=executed_patch_steps, intervention_action_steps=intervened_steps)
    return row, events


def run_evaluation(config: EvaluationConfig, adapter: LiberoAdapter, output, *, shard: Shard = Shard(),
                   max_new_episodes: int | None = None, trace: Callable | None = None,
                   on_episode: Callable | None = None) -> dict:
    if adapter.config != config:
        raise ValueError("adapter and evaluation configurations differ")
    if max_new_episodes is not None and (type(max_new_episodes) is not int or max_new_episodes < 1):
        raise ValueError("max_new_episodes must be positive")
    with ResultStore(output, config, shard) as store:
        missing = set(shard.rollout_ids) - store.completed
        completed_now = 0
        for task_id in range(10):
            ids = sorted(i for i in missing if i // 50 == task_id)
            if not ids:
                continue
            try:
                adapter.open_task(task_id, ids[0] % 50)
                for rid in ids:
                    row, events = run_episode(adapter, rid, trace=trace)
                    store.commit(row, events)
                    if on_episode:
                        on_episode(store.rows[-1], store.summary)
                    completed_now += 1
                    if max_new_episodes is not None and completed_now >= max_new_episodes:
                        return store.summary
            finally:
                adapter.close_task()
        return store.summary
