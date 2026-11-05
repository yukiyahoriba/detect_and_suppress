"""The research SmolVLA LIBERO camera, preprocessing and RNG boundary."""

from __future__ import annotations

import numpy as np

from ..evaluation.rollout import Prediction
from ..evaluation.patches import overlay_patch_on_image
from .libero_support import initial_states, load_evaluation_defense, load_suite, validate_libero_identity
from .smolvla import SmolVLADefense


def patch_observation(observation, patch, spec, rollout_id):
    result = {**observation, "pixels": dict(observation["pixels"])}
    for camera, key in enumerate(("image", "image2")):
        frame = np.ascontiguousarray(observation["pixels"][key][::-1, ::-1])
        frame = np.asarray(overlay_patch_on_image(frame, patch, spec=spec, episode_id=rollout_id,
                                                timestep=0, camera_index=camera), dtype=np.uint8)
        result["pixels"][key] = np.ascontiguousarray(frame[::-1, ::-1])
    return result


class SmolVLALiberoAdapter:
    """Delegate model inference to SmolVLADefense and environment steps to LeRobot."""

    flush_on_patch_change = False

    def __init__(self, config, *, device="cuda"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.random_utils import set_seed
        from PIL import Image

        if config.model != "smolvla":
            raise ValueError("expected SmolVLA evaluation config")
        if not torch.cuda.is_available() or not device.startswith("cuda"):
            raise ValueError("paper SmolVLA evaluation uses CUDA policy noise")
        validate_libero_identity(config)
        bundle = load_evaluation_defense(config)
        self.config, self.env, self.device = config, None, device
        set_seed(config.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        checkpoint = config.artifacts["checkpoint"].path
        policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        policy_config.device, policy_config.num_steps = device, config.denoising_steps
        if getattr(policy_config, "rtc_config", None) is not None:
            raise ValueError("RTC inference is not part of the paper evaluation")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_config,
                                                   local_files_only=True, strict=True).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config, pretrained_path=checkpoint)
        env_config = LiberoEnvConfig(task=config.task_suite, observation_height=256,
                                    observation_width=256, episode_length=config.max_steps)
        self.env_preprocessor, _ = make_env_pre_post_processors(env_config, policy_config)
        self.suite = load_suite()
        for task_id in range(10):
            initial_states(self.suite, task_id)
        self.patch = Image.open(config.artifacts["patch"].path).convert("RGBA") if config.patch_mode != "none" else None
        self.defense = SmolVLADefense(self.policy.model, bundle, checkpoint=checkpoint)

    def open_task(self, task_id, first_init):
        from lerobot.envs.libero import LiberoEnv

        self.close_task()
        self.task_id = task_id
        self.task_language = str(self.suite.get_task(task_id).language)
        self.env = LiberoEnv(task_suite=self.suite, task_id=task_id, task_suite_name=self.config.task_suite,
                            episode_length=self.config.max_steps, obs_type="pixels_agent_pos",
                            observation_width=256, observation_height=256, episode_index=first_init,
                            num_steps_wait=self.config.num_steps_wait, control_mode="relative", use_patch=False)

    def reset_episode(self, rollout_id):
        import torch

        if rollout_id // 50 != self.task_id:
            raise ValueError("episode is outside the open task")
        torch.manual_seed(self.config.seed + rollout_id)
        torch.cuda.manual_seed_all(self.config.seed + rollout_id)
        self.env.init_state_id = rollout_id % 50
        obs, _ = self.env.reset(seed=self.config.seed)
        for processor in (self.preprocessor, self.postprocessor, self.env_preprocessor):
            processor.reset()
        self.defense.reset_episode(rollout_id)
        return obs

    def prepare_observation(self, observation, patch_active, rollout_id, timestep):
        return patch_observation(observation, self.patch, self.config.overlay(), rollout_id) if patch_active else observation

    def predict(self, observation, patch_active):
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

        inference = {**observation, "robot_state": {
            group: {name: np.asarray(value)[None, ...] for name, value in entries.items()}
            for group, entries in observation["robot_state"].items()}}
        processed = preprocess_observation(inference)
        processed["task"] = [self.task_language]
        processed = self.preprocessor(self.env_preprocessor(processed))
        policy_config = self.policy.config
        # Draw directly on the inference device from the per-episode global RNG.
        noise = torch.randn((1, policy_config.chunk_size, policy_config.max_action_dim),
                            dtype=torch.float32, device=self.device)
        images, masks, streams = self.policy.prepare_images(processed, return_streams=True)
        state = self.policy.prepare_state(processed)
        kwargs = {"oracle_active": bool(patch_active)} if self.config.defense.mode == "oracle" else {}
        with torch.inference_mode():
            normalized = self.defense.sample_actions(images, masks, processed[OBS_LANGUAGE_TOKENS],
                processed[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise, image_streams=streams, **kwargs)
            predicted = self.postprocessor(normalized[:, :, :policy_config.action_feature.shape[0]])
        record = self.defense.last_record
        return Prediction(np.asarray(predicted.detach().cpu().numpy()[0], dtype=np.float32),
                          float(record.scores[0]), bool(record.detected[0]), bool(record.applied[0]),
                          record.intervention_calls, normalized)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, bool(terminated or truncated), bool(info.get("is_success", False))

    def close_task(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def close(self):
        self.close_task()
        self.defense.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
