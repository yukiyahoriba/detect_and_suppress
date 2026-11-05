"""SmolVLA hooks around the existing prefix and action-generation methods."""

from __future__ import annotations

from contextlib import contextmanager
import threading

import numpy as np
import torch

from ..backends.torch import suppress_features
from ..defense import DefenseBundle, DefenseHistory, DefenseRecord, check_checkpoint, validate_mode


class SmolVLADefense:
    """Attach defense to a loaded, eval-mode SmolVLA flow model.

    ``sample_actions`` delegates to the model's existing method. ``forward``
    also supports existing instrumented samplers, including their diagnostic
    pass. The adapter owns no VLA weights and never implements denoising.
    """

    def __init__(self, flow, defense: DefenseBundle, *, checkpoint=None, capture_activations=False,
                 image_streams=("agent", "wrist"), history_size=128):
        check_checkpoint(defense, checkpoint)
        config = defense.config
        if not config.detector.model.startswith("smolvla") or config.intervention.family != "vlm":
            raise ValueError("SmolVLA intervention requires a VLM residual stream")
        if config.intervention.capture_point != "VLM block input_layernorm input (pre-RMSNorm residual)":
            raise ValueError("unsupported intervention capture point")
        if config.detector.family == "vlm":
            if config.detector.capture_point != config.intervention.capture_point or config.detector.pooling != "FP32 masked mean over all valid prefix tokens":
                raise ValueError("unsupported VLM detector capture/pooling")
            if config.detector.layer > config.intervention.layer:
                raise ValueError("detector must precede the intervention in the same prefix")
        elif config.detector.family == "vit":
            if config.detector.capture_point != "SmolVLM vision encoder block output (post-block, before final post_layernorm)" or config.detector.pooling != "FP32 mean over image tokens per camera, concatenated as [agent; wrist]":
                raise ValueError("unsupported ViT detector capture/pooling")
        else:
            raise ValueError("detector must use a VLM or ViT residual")
        if getattr(flow, "training", False):
            raise ValueError("put the loaded model in eval mode before attaching defense")
        if getattr(flow, "_detect_and_suppress_owner", None) is not None:
            raise ValueError("this flow already has a defense adapter")
        self.flow, self.defense = flow, defense
        self.capture_activations = capture_activations
        self.image_streams = tuple(image_streams)
        if not self.image_streams or (config.detector.family == "vit" and self.image_streams != ("agent", "wrist")):
            raise ValueError("the legacy ViT Probe expects agent then wrist camera calls")
        if config.detector.image_keys and config.detector.image_keys != self.image_streams:
            raise ValueError("Probe camera order differs from image_streams")
        self._history = DefenseHistory(history_size)
        self._lock = threading.Lock()
        self._active, self._closed, self._phase = False, False, None
        self._handles, self._overrides = [], []
        self._clear()
        raw_direction = torch.as_tensor(defense.decoder_direction, dtype=torch.float32)
        self._direction = raw_direction / torch.linalg.vector_norm(raw_direction).clamp_min(1e-8)
        self._probe = {key: torch.tensor(np.asarray(getattr(defense.probe, key)).copy(), dtype=torch.float32)
                       for key in ("mean", "scale", "coef", "intercept", "threshold")}
        self._intercept = float(self._probe["intercept"])
        self._threshold = float(self._probe["threshold"])
        vlm = flow.vlm_with_expert.get_vlm_model()
        layers = vlm.text_model.layers
        try:
            if config.intervention.layer >= len(layers) or (config.detector.family == "vlm" and config.detector.layer >= len(layers)):
                raise ValueError("residual layer is outside the loaded SmolVLA model")
            target = layers[config.intervention.layer].input_layernorm
            weight = getattr(target, "weight", None)
            if weight is not None and weight.numel() != config.intervention.hidden_dim:
                raise ValueError("loaded VLM dimension differs from the intervention representation")
            if config.detector.family == "vlm":
                detector = layers[config.detector.layer].input_layernorm
                self._same = detector is target
                self._handles.append(detector.register_forward_pre_hook(self._detect_vlm))
            else:
                self._same = False
                if config.detector.layer >= len(vlm.vision_model.encoder.layers):
                    raise ValueError("ViT layer is outside the loaded model")
                vision = vlm.vision_model.encoder.layers[config.detector.layer]
                self._handles.append(vision.register_forward_hook(self._detect_vit))
            if not self._same:
                self._handles.append(target.register_forward_pre_hook(self._intervene))
            self._override(flow, "embed_prefix", self._wrap_embed(flow.embed_prefix))
            self._override(flow.vlm_with_expert, "forward", self._wrap_forward(flow.vlm_with_expert.forward))
            flow._detect_and_suppress_owner = self
        except BaseException:
            self.close()
            raise

    def _clear(self):
        self._scores = self._detected = self._gate = self._prefix_mask = None
        self._before = self._after = None
        self._vision = []
        self._detections = self._targets = self._prefixes = 0

    def _override(self, owner, name, wrapper):
        self._overrides.append((owner, name, name in owner.__dict__, owner.__dict__.get(name), wrapper))
        setattr(owner, name, wrapper)

    def _wrap_embed(self, original):
        def wrapped(*args, **kwargs):
            if not self._active:
                return original(*args, **kwargs)
            if self._prefixes or self._prefix_mask is not None:
                raise RuntimeError("one managed forward must contain exactly one prefix embedding")
            streams = kwargs.get("image_streams")
            if streams is not None and tuple(streams) != self.image_streams:
                raise ValueError("image camera order differs from the configured order")
            self._phase = "embedding"
            try:
                result = original(*args, **kwargs)
                self._prefix_mask = result[1]
                return result
            finally:
                self._phase = None
        return wrapped

    def _wrap_forward(self, original):
        def wrapped(*args, **kwargs):
            embeddings = kwargs.get("inputs_embeds")
            prefix = embeddings is not None and embeddings[0] is not None and embeddings[1] is None
            if not self._active or not prefix:
                return original(*args, **kwargs)
            if self._prefix_mask is None or self._prefixes:
                raise RuntimeError("prefix must follow one embedding pass in the current forward")
            self._prefixes += 1
            self._phase = "prefix"
            try:
                return original(*args, **kwargs)
            finally:
                self._phase = None
        return wrapped

    def _score(self, pooled):
        if pooled.shape[-1] != self.defense.config.detector.hidden_dim or self._detections:
            raise RuntimeError("invalid detector dimension or repeated detection")
        self._scores = ((pooled - self._probe["mean"].to(pooled.device)) /
                        self._probe["scale"].to(pooled.device)) @ self._probe["coef"].to(pooled.device) + self._intercept
        self._detected = self._scores > self._threshold
        self._detections += 1

    def _detect_vlm(self, _module, inputs):
        if not self._active or self._phase != "prefix":
            return None
        hidden, mask = inputs[0], self._prefix_mask
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
            raise RuntimeError("prefix token mask does not align with residuals")
        weighted = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        pooled = torch.sum(weighted, dim=1, dtype=torch.float32) / torch.sum(
            mask, dim=1, dtype=torch.float32).clamp_min(1.).unsqueeze(-1)
        self._score(pooled)
        return self._apply(inputs) if self._same else None

    def _detect_vit(self, _module, _inputs, output):
        if not self._active or self._phase != "embedding":
            return
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise RuntimeError("expected ViT token residuals")
        self._vision.append(hidden.float().mean(dim=1).detach())
        if len(self._vision) > len(self.image_streams):
            raise RuntimeError("too many vision calls in one prefix")
        if len(self._vision) == len(self.image_streams):
            self._score(torch.cat(self._vision, dim=-1))

    def _intervene(self, _module, inputs):
        if self._active and self._phase == "prefix":
            return self._apply(inputs)
        return None

    def _apply(self, inputs):
        if self._detected is None or self._targets:
            raise RuntimeError("intervention must follow exactly one detection")
        hidden = inputs[0]
        if hidden.shape[-1] != self.defense.config.intervention.hidden_dim:
            raise RuntimeError("intervention residual dimension differs from the SAE")
        if self._mode == "probe":
            gate = self._detected
        elif self._mode == "always":
            gate = torch.ones_like(self._detected)
        elif self._mode == "disabled":
            gate = torch.zeros_like(self._detected)
        else:
            gate = torch.as_tensor(self._oracle, device=hidden.device, dtype=torch.bool)
            if gate.ndim == 0:
                gate = gate.expand(hidden.shape[0])
            if gate.shape != (hidden.shape[0],):
                raise ValueError("oracle_active must be scalar or [batch]")
        self._gate = gate
        kernel_gate = gate
        if self._mode in ("always", "disabled"):
            kernel_gate = self._mode == "always"
        elif self._mode == "oracle" and isinstance(self._oracle, (bool, np.bool_)):
            kernel_gate = bool(self._oracle)
        updated = suppress_features(hidden, self._direction, lambda_=self.defense.config.lambda_,
                                     active=kernel_gate, normalize_direction=False)
        if self.capture_activations:
            self._before, self._after = hidden.detach().clone(), updated.detach().clone()
        self._targets += 1
        return (updated, *inputs[1:])

    @contextmanager
    def forward(self, *, mode=None, oracle_active=None):
        """Manage one call to the existing sampler, including cleanup on exceptions."""
        if self._closed or not self._lock.acquire(blocking=False):
            raise RuntimeError("adapter is closed or already running a forward")
        try:
            self._mode = mode if mode is not None else self.defense.config.mode
            validate_mode(self._mode)
            if self._mode == "oracle" and oracle_active is None:
                raise ValueError("oracle mode requires explicit oracle_active")
            if self._mode != "oracle" and oracle_active is not None:
                raise ValueError("oracle_active is only valid in oracle mode")
            self._clear()
            self._history.last_record = None
            self._oracle = oracle_active
            self._active = True
            yield self
            if (self._prefixes, self._detections, self._targets) != (1, 1, 1):
                raise RuntimeError("forward did not execute one prefix, detection and intervention target")
            def host(value):
                result = value.detach().cpu().numpy().copy()
                result.setflags(write=False)
                return result
            self._history.append(DefenseRecord(
                self._history.episode_id, self._history.forward_id, self._mode,
                self.defense.config.detector.layer, self.defense.config.intervention.layer,
                host(self._scores), host(self._detected), host(self._gate),
                int(self._gate.sum().item()), "prefix", self._before, self._after,
            ))
        finally:
            self._active, self._phase = False, None
            self._clear()
            self._lock.release()

    def sample_actions(self, *args, mode=None, oracle_active=None, **kwargs):
        if kwargs.get("image_streams") is not None and tuple(kwargs["image_streams"]) != self.image_streams:
            raise ValueError("image camera order differs from the configured order")
        with self.forward(mode=mode, oracle_active=oracle_active), torch.inference_mode():
            return self.flow.sample_actions(*args, **kwargs)

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
            self._clear()
            self._history.reset(episode_id)
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an active forward")
        try:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            for owner, name, existed, previous, wrapper in reversed(self._overrides):
                if getattr(owner, name) is wrapper:
                    if existed:
                        setattr(owner, name, previous)
                    else:
                        delattr(owner, name)
            self._overrides.clear()
            if getattr(self.flow, "_detect_and_suppress_owner", None) is self:
                delattr(self.flow, "_detect_and_suppress_owner")
            self._clear()
            self._history.reset(None)
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("adapter is closed")
        return self

    def __exit__(self, *_):
        self.close()
