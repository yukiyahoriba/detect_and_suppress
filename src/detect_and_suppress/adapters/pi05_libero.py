"""The research Pi0.5 agent-only LIBERO observation and action boundary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..evaluation.rollout import Prediction
from ..evaluation.patches import overlay_patch_on_image
from .libero_support import initial_states, load_evaluation_defense, load_suite, validate_libero_identity
from .pi05 import Pi05Defense


class AgentOnlyImages:
    def __call__(self, data):
        image = data.get("image")
        if not isinstance(image, dict) or image.get("base_0_rgb") is None:
            return data
        zero = np.zeros_like(image["base_0_rgb"])
        image["left_wrist_0_rgb"] = zero
        image["right_wrist_0_rgb"] = zero
        if isinstance(data.get("image_mask"), dict):
            data["image_mask"]["left_wrist_0_rgb"] = np.False_
            data["image_mask"]["right_wrist_0_rgb"] = np.False_
        return data


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - quat[3] * quat[3])
    if math.isclose(float(den), 0.):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2. * math.acos(float(quat[3])) / den).astype(np.float32)


def unbatch(value):
    if isinstance(value, (str, bytes)):
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array[0] if array.shape[0] == 1 else array


class Pi05LiberoAdapter:
    """Load native OpenPI transforms and invoke Pi05Defense for each replan."""

    flush_on_patch_change = True

    def __init__(self, config):
        import jax
        import jax.numpy as jnp
        from openpi import transforms
        from openpi.models import model as model_api
        from openpi.training import checkpoints, config as training_config
        from PIL import Image

        if config.model != "pi05":
            raise ValueError("expected Pi0.5 evaluation config")
        validate_libero_identity(config)
        bundle = load_evaluation_defense(config)
        if bundle.config.detector.image_keys and bundle.config.detector.image_keys != ("base_0_rgb",):
            raise ValueError("paper Pi0.5 evaluation uses only the agent camera")
        self.config, self.env = config, None
        self.suite = load_suite()
        self.states = [initial_states(self.suite, i) for i in range(10)]
        checkpoint = Path(config.artifacts["checkpoint"].path)
        train = training_config.get_config(config.config_name)
        model = train.model.load(model_api.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        data = train.data.create(checkpoint / "assets", train.model)
        stats = checkpoints.load_norm_stats(checkpoint / "assets", data.asset_id)
        self.input_transform = transforms.compose([
            transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs, AgentOnlyImages(),
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs])
        self.output_transform = transforms.compose([
            *data.model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs])
        self.patch = Image.open(config.artifacts["patch"].path).convert("RGBA") if config.patch_mode != "none" else None
        self.base_rng = jax.random.key(config.seed)
        self.defense = Pi05Defense(model, bundle, checkpoint=str(checkpoint), timing=config.timing)

    def open_task(self, task_id, first_init):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.close_task()
        self.task_id = task_id
        task = self.suite.get_task(task_id)
        self.task_language = str(task.language)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        self.env.seed(self.config.seed)

    def reset_episode(self, rollout_id):
        import jax

        if rollout_id // 50 != self.task_id:
            raise ValueError("episode is outside the open task")
        self.env.reset()
        obs = self.env.set_init_state(self.states[self.task_id][rollout_id % 50])
        for _ in range(self.config.num_steps_wait):
            obs, _, _, _ = self.env.step([0.] * 6 + [-1.])
        self.rng = jax.random.fold_in(self.base_rng, rollout_id)
        self.defense.reset_episode(rollout_id)
        return obs

    def prepare_observation(self, observation, patch_active, rollout_id, timestep):
        from openpi_client import image_tools

        image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            image, self.config.resize_size, self.config.resize_size))
        if patch_active:
            image = np.asarray(overlay_patch_on_image(image, self.patch, spec=self.config.overlay(),
                episode_id=rollout_id, timestep=timestep, camera_index=0), dtype=np.uint8)
        return {"observation/image": image, "observation/wrist_image": np.zeros_like(image),
                "observation/state": np.concatenate((observation["robot0_eef_pos"],
                    quat2axisangle(observation["robot0_eef_quat"]), observation["robot0_gripper_qpos"])),
                "prompt": self.task_language}

    def predict(self, observation, patch_active):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api

        inputs = self.input_transform(dict(observation))
        inputs = jax.tree.map(lambda v: jnp.asarray(v)[np.newaxis, ...], inputs)
        obs = model_api.Observation.from_dict(inputs)
        self.rng, sample_rng = jax.random.split(self.rng)
        kwargs = {"oracle_active": bool(patch_active)} if self.config.defense.mode == "oracle" else {}
        actions = self.defense.sample_actions(sample_rng, obs, num_steps=self.config.denoising_steps, **kwargs)
        outputs = self.output_transform({"state": inputs["state"], "actions": actions})
        outputs = jax.tree.map(unbatch, outputs)
        chunk = np.asarray(outputs["actions"], dtype=np.float32)
        if chunk.ndim == 2 and chunk.shape[1] > 7:
            chunk = chunk[:, :7]
        record = self.defense.last_record
        return Prediction(chunk, float(record.scores[0]), bool(record.detected[0]), bool(record.applied[0]),
                          record.intervention_calls, actions)

    def step(self, action):
        obs, _, done, _ = self.env.step(action.tolist())
        return obs, bool(done), bool(done)

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
