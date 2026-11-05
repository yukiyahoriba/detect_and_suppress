"""Pi0.5 defense through the existing OpenPI sampling methods."""

from __future__ import annotations

import inspect
import threading

import jax
import jax.numpy as jnp
import numpy as np

from ..defense import DefenseBundle, DefenseHistory, DefenseRecord, check_checkpoint, validate_mode


class Pi05Defense:
    """Use the prepared OpenPI overlay with explicit prefix/denoising timing.

    Prefix timing is the legacy Probe-gated path. Denoising timing delegates
    to the legacy repeated-prefix path used for Oracle/always experiments.
    No action integration loop is implemented here.
    """

    def __init__(self, model, defense: DefenseBundle, *, checkpoint=None, timing="prefix",
                 capture_activations=False, history_size=128, jit=True):
        check_checkpoint(defense, checkpoint)
        spec, target = defense.config.detector, defense.config.intervention
        if not spec.model.startswith("pi05") or target.family not in ("llm", "pg"):
            raise ValueError("Pi0.5 intervention requires the PaliGemma residual stream")
        if target.capture_point != "PaliGemma transformer block input residual stream":
            raise ValueError("unsupported intervention capture point")
        if not 0 <= target.layer < 18:
            raise ValueError("invalid PaliGemma intervention layer")
        if timing not in ("prefix", "denoising"):
            raise ValueError("timing must be prefix or denoising")
        if spec.family in ("llm", "pg"):
            if spec.capture_point != target.capture_point or spec.pooling != "unmasked arithmetic mean over all PaliGemma prefix tokens":
                raise ValueError("unsupported PaliGemma detector capture/pooling")
            if not 0 <= spec.layer <= target.layer or spec.hidden_dim != target.hidden_dim:
                raise ValueError("PaliGemma detector must precede intervention in the same residual stream")
            method = model.sample_actions_with_probe_gated_sae_steering
        elif spec.family == "vit":
            if spec.capture_point != "SigLIP Encoder1DBlock +mlp output, before the next block LayerNorm" or spec.pooling != "FP32 arithmetic mean over spatial patch tokens; cameras concatenated":
                raise ValueError("unsupported SigLIP detector capture/pooling")
            if not 0 <= spec.layer < 27 or not spec.image_keys or spec.hidden_dim != 1152 * len(spec.image_keys):
                raise ValueError("SigLIP detector requires explicit camera order and matching dimension")
            if timing != "prefix":
                raise ValueError("the legacy repeated-prefix path supports only a PaliGemma detector")
            method = model.sample_actions_with_vit_probe_gated_sae_steering
        else:
            raise ValueError("unsupported Pi0.5 detector family")
        if timing == "denoising":
            method = model.sample_actions_with_sae_steering_and_activation
        if "defense_mode" not in inspect.signature(method).parameters:
            raise ValueError("load OpenPI from prepare_openpi_overlay before constructing the model")
        if not getattr(model, "pi05", False):
            raise ValueError("the loaded model is not Pi0.5")
        native_module = getattr(model.PaliGemma.llm, "module", None)
        configs = getattr(native_module, "configs", None)
        if configs is not None and configs[0].width != target.hidden_dim:
            raise ValueError("loaded PaliGemma dimension differs from the SAE representation")
        static = ["layer", "expert", "direction", "mode_id", "alpha", "num_steps", "defense_mode", "capture_defense"]
        if timing == "denoising":
            static.append("defense_detector_layer")
        else:
            static.append("detector_layer")
            if spec.family == "vit":
                static.append("defense_image_keys")
        if jit:
            from openpi.shared import nnx_utils

            method = nnx_utils.module_jit(method, static_argnames=tuple(static))
        self.model, self.defense, self.timing = model, defense, timing
        self.capture_activations = capture_activations
        self._sample = method
        self._history = DefenseHistory(history_size)
        self._lock = threading.Lock()
        self._closed = False
        self._probe = {key: jnp.asarray(getattr(defense.probe, key), dtype=jnp.float32)
                       for key in ("mean", "scale", "coef", "intercept", "threshold")}
        self._direction = tuple(float(value) for value in defense.direction)

    def sample_actions(self, rng, observation, *, noise=None, num_steps=10, mode=None, oracle_active=None):
        if self._closed or not self._lock.acquire(blocking=False):
            raise RuntimeError("adapter is closed or already running a forward")
        try:
            self._history.last_record = None
            mode = mode if mode is not None else self.defense.config.mode
            validate_mode(mode)
            if self.timing == "denoising" and mode == "probe":
                raise ValueError("Probe gating uses prefix timing; repeated-prefix mode is for Oracle/always/disabled")
            if not isinstance(num_steps, int) or num_steps < 1:
                raise ValueError("num_steps must be a positive integer")
            batch = observation.state.shape[0]
            if mode == "oracle":
                if oracle_active is None:
                    raise ValueError("oracle mode requires explicit oracle_active")
                active = jnp.asarray(oracle_active, dtype=jnp.bool_)
                if active.shape not in ((), (batch,)):
                    raise ValueError("oracle_active must be scalar or [batch]")
                active = jnp.broadcast_to(active, (batch,))
            else:
                if oracle_active is not None:
                    raise ValueError("oracle_active is only valid in oracle mode")
                active = jnp.zeros((batch,), dtype=jnp.bool_)
            spec = self.defense.config.detector
            kwargs = dict(layer=self.defense.config.intervention.layer, expert=0, direction=self._direction,
                          mode_id=3, alpha=self.defense.config.lambda_, num_steps=num_steps, noise=noise,
                          defense_mode=mode, defense_active=active, capture_defense=self.capture_activations)
            if self.timing == "denoising":
                actions, extras = self._sample(rng, observation, defense_probe=self._probe,
                                               defense_detector_layer=spec.layer, **kwargs)
            else:
                if spec.family == "vit":
                    kwargs["defense_image_keys"] = spec.image_keys
                actions, extras = self._sample(rng, observation, self._probe["mean"], self._probe["scale"],
                                               self._probe["coef"], self._probe["intercept"], self._probe["threshold"],
                                               detector_layer=spec.layer, **kwargs)
            data = extras["defense"]
            def host(value):
                result = np.asarray(jax.device_get(value)).copy()
                result.setflags(write=False)
                return result
            scores, detected, applied = host(extras["probe"]["score"]), host(extras["probe"]["detected"]), host(data["applied"])
            self._history.append(DefenseRecord(
                self._history.episode_id, self._history.forward_id, mode, spec.layer,
                self.defense.config.intervention.layer, scores, detected, applied,
                int(applied.sum()) * (num_steps if self.timing == "denoising" else 1), self.timing,
                data["before"] if self.capture_activations else None,
                data["after"] if self.capture_activations else None,
            ))
            return actions
        finally:
            self._lock.release()

    @property
    def last_record(self):
        return self._history.last_record

    @property
    def history(self):
        return self._history.history

    def reset_episode(self, episode_id=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot reset an active forward")
        try:
            self._history.reset(episode_id)
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an active forward")
        try:
            self._history.reset(None)
            self._closed = True
            self._sample = None
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("adapter is closed")
        return self

    def __exit__(self, *_):
        self.close()
