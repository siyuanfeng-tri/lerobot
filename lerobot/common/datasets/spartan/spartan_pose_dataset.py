import copy
import functools
import os
from typing import Dict, List, Optional

import cv2
import numcodecs
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
from tqdm import tqdm
import yaml

from lerobot.common.datasets.spartan.fsspec_util import (
    load_file_with_fsspec,
    load_npz_with_fsspec,
)
from lerobot.common.datasets.spartan.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
)
from lerobot.common.datasets.spartan.parallel_work import parallel_work
from lerobot.common.datasets.spartan.path_util import resolve_glob_type_to_list
from lerobot.common.datasets.spartan.pytorch_util import dict_apply, dict_apply_reduce
from lerobot.common.datasets.spartan.replay_buffer import ReplayBuffer
from lerobot.common.datasets.spartan.sampler import SequenceSampler, get_val_mask
from lerobot.common.datasets.spartan.base_dataset import BaseImageDataset
from lerobot.common.datasets.spartan.relative_trajectory_conversion import (
    change_to_relative_trajectories,
)
from lerobot.common.datasets.spartan.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from lerobot.common.datasets.spartan.rotation_transformer import (
    get_rotation_transformer,
)


class SpartanBaseDataset(BaseImageDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        mode="np",
        compressor="blosc",
        max_num_episodes: Optional[int] = None,
        raw_rgb=True,
        num_workers=8,
        replay_buffer_path=None,
        is_multiarm=False,
        is_relative=False,
        # TODO(sfeng): should lift this into resolve_glob_type_to_list()
        path_is_fully_resolved: bool = False,
    ):
        if is_relative:
            assert (
                is_multiarm
            ), "Relative trajectories not supported for single arm data."
        # hack..
        assert rotation_rep in ["rotation_6d", "pitch"]

        obs_shape_meta = shape_meta["obs"]

        self._is_multiarm = is_multiarm
        self._camera_names = []
        self._lowdim_names = []
        self._raw_rgb = raw_rgb
        self._shape_meta = shape_meta
        self._is_relative = is_relative

        print(f"\n\nIs relative traj: {self._is_relative}\n")
        # fail fast check
        for arm in ["left", "right"]:
            other_arm = "right" if arm == "left" else "left"
            if self._is_relative:
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__xyz"
                    in obs_shape_meta
                )
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__rot_6d"
                    in obs_shape_meta
                )
            else:
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__xyz"
                    not in obs_shape_meta
                )
                assert (
                    f"robot__actual__poses__{arm}__{other_arm}::panda__rot_6d"
                    not in obs_shape_meta
                )

        # camera_name: (h, w)
        self._image_shapes = {}
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])

            # type defaults to "low_dim"
            type = attr.get("type", "low_dim")

            if type == "rgb":
                self._camera_names.append(key)
                assert len(shape) == 3

                self._image_shapes[key] = shape[1:]

            elif type == "low_dim":
                self._lowdim_names.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # idk how to deal with this correctly
        if isinstance(episode_path_globs, ListConfig):
            episode_path_globs = list(episode_path_globs)

        if path_is_fully_resolved:
            episode_paths = episode_path_globs
        else:
            episode_paths = resolve_glob_type_to_list(episode_path_globs)

        if max_num_episodes is not None and (
            max_num_episodes < len(episode_paths)
        ):
            episode_paths = episode_paths[:max_num_episodes]
            assert len(episode_paths) == max_num_episodes

        if mode == "np" or replay_buffer_path is None:
            # load all episodes
            print(f"Building replay buffer from scratch with stride: {stride}")
            replay_buffer = make_replay_buffer(
                episode_paths=episode_paths,
                rotation_rep=rotation_rep,
                repeat_head=repeat_head,
                repeat_tail=repeat_tail,
                mode=mode,
                compressor=compressor,
                num_workers=num_workers,
                stride=stride,
                is_multiarm=is_multiarm,
                camera_names=self._camera_names,
                lowdim_names=self._lowdim_names,
                image_shapes=self._image_shapes,
                has_gripper=has_gripper,
            )
        else:
            print(f"Loading replay buffer from: {replay_buffer_path}")
            if replay_buffer_path.startswith("s3://"):
                replay_buffer = ReplayBuffer.create_from_s3(replay_buffer_path)
            else:
                replay_buffer = ReplayBuffer.create_from_path(
                    replay_buffer_path
                )

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.replay_buffer = replay_buffer
        self.sampler = sampler

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @property
    def num_samples(self):
        return len(self)

    @property
    def num_episodes(self):
        return len(self.replay_buffer.episode_ends)


class EpisodePathHelper:
    def __init__(self, episode_path: str) -> None:
        self.episode_path = episode_path
        self._base_path = os.path.join(self.episode_path, "processed")

    @property
    def actions_path(self) -> str:
        path = os.path.join(self._base_path, "actions.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path

    @property
    def observations_path(self) -> str:
        path = os.path.join(self._base_path, "observations.npz")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path

    @property
    def metadata_path(self) -> str:
        path = os.path.join(self._base_path, "metadata.yaml")
        assert os.path.exists(path), f"{path} doesn't exist."
        return path


def make_replay_buffer(
    *,
    episode_paths: List[str],
    rotation_rep: str,
    repeat_head: int,
    repeat_tail: int,
    mode: str,
    compressor: str,
    num_workers: int,
    stride: int,
    is_multiarm: bool,
    camera_names: List[str],
    lowdim_names: List[str],
    image_shapes: Dict[str, tuple],
    has_gripper: Optional[bool],
):
    assert mode in ["np", "zarr"], f"Unknown storage mode: {mode}"
    assert compressor in [
        "blosc",
        "jpeg2k",
    ], f"Unknown compressor: {compressor}"

    if mode == "np":
        replay_buffer = ReplayBuffer.create_empty_numpy()
    elif mode == "zarr":
        replay_buffer = ReplayBuffer.create_empty_zarr()

    if compressor == "blosc":
        compressor = numcodecs.Blosc(
            cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
        )

    if not is_multiarm:
        rotation_transformer = get_rotation_transformer(
            from_rep="axis_angle",
            to_rep=rotation_rep,
        )

    def worker(paths):
        for episode_path in paths:
            # load the obj and action trajs.
            path_helper = EpisodePathHelper(episode_path)

            observations = load_npz_with_fsspec(
                path_helper.observations_path,
            )
            actions = load_npz_with_fsspec(
                path_helper.actions_path,
            )
            metadata = load_file_with_fsspec(
                path_helper.metadata_path,
                yaml.safe_load,
            )

            assert (
                "camera_id_to_semantic_name" in metadata
            ), f"Missing camera_id_to_semantic_name in {episode_path}"
            camera_id_to_camera_name = metadata["camera_id_to_semantic_name"]

            # Confirm that there is exactly one skill represented.
            # TODO(charlie): Handle multiple skills in a rollout.
            assert len(metadata["skills"].items()) == 1
            skill_name = next(iter(metadata["skills"].keys()))

            if is_multiarm:
                episode = bimanual_spartan_episode_to_replay_buffer_episode(
                    observation_dict=observations,
                    action_dict=actions,
                    skill_name=skill_name,
                    lowdim_keys=lowdim_names,
                    camera_names=camera_names,
                    camera_id_to_camera_name=camera_id_to_camera_name,
                    repeat_head=repeat_head,
                    repeat_tail=repeat_tail,
                    image_shapes=image_shapes,
                )
            else:
                episode = spartan_episode_to_replay_buffer_episode(
                    observation_dict=observations,
                    action_dict=actions,
                    skill_name=skill_name,
                    lowdim_keys=lowdim_names,
                    camera_names=camera_names,
                    camera_id_to_camera_name=camera_id_to_camera_name,
                    rotation_transformer=rotation_transformer,
                    repeat_head=repeat_head,
                    repeat_tail=repeat_tail,
                    has_gripper=has_gripper,
                    image_shapes=image_shapes,
                )
            # we take stride > 1 to downsample in the time dimension.
            # this only takes effect when making a new replay buffer from
            # spartan data.
            if stride != 1:
                strided = {}
                for k, v in episode.items():
                    # T should be the first dim.
                    index = np.arange(0, v.shape[0], step=stride)
                    strided[k] = v[index]
                episode = strided

            yield (episode, episode_path)

    episodes = parallel_work(
        worker,
        episode_paths,
        process_count=num_workers,
        progress_cls=functools.partial(
            tqdm, desc=f"Loading episodes w. {num_workers} proc"
        ),
    )

    for episode, path in tqdm(episodes, desc="Adding to ReplayBuffer"):
        # add image compressors, we will the vector ones.
        compressors = {}
        chunks = {}
        for cam_name in camera_names:
            name = f"obs.{cam_name}"
            compressors[name] = compressor
            _, h, w, c = episode[name].shape
            chunks[name] = (1, h, w, c)
        replay_buffer.add_episode(
            episode,
            path=path,
            # compressors arg have no effect for np backend.
            compressors=compressors,
            chunks=chunks,
        )

    return replay_buffer


def get_normalizer_params(x, output_max=1, output_min=-1, range_eps=1e-7):
    N, d = x.shape
    # 4 is for pitch
    # assert d in [9, 10, 1, 4]
    # assume always [xyz, 6d_rot, maybe_gripper]
    # assume always [xyz, 1d_pitch, maybe_gripper]
    input_min = np.min(x, axis=0)
    input_max = np.max(x, axis=0)
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps

    # set rotation dims to always ignore, since these are always in -1 to 1
    # if d != 1:
    #    ignore_dim[3:9] = True

    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    # scale[ignore_dim] = 1
    offset = output_min - scale * input_min
    # offset[ignore_dim] = (
    #     (output_max + output_min) / 2 - input_min[ignore_dim]
    # )
    offset[ignore_dim] = 0

    info = {
        "min": input_min,
        "max": input_max,
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
    }

    return scale, offset, info


def _change_pose_repr(poses, rotation_transformer):
    N, d = poses.shape
    assert N >= 1

    # assuming gripper width...
    if d == 1:
        return poses.astype(np.float32)

    assert d == 6
    # assuming xyz, axang
    rot = poses[:, 3:]
    pos = poses[:, :3]
    new_rot = rotation_transformer.forward(rot)

    result = np.concatenate([pos, new_rot], axis=-1).astype(np.float32)
    return result


def _repeat(x, num, mode="head"):
    assert num >= 0
    if mode == "head":
        stuff = [x[:1] for i in range(num)]
    else:
        stuff = [x[-1:] for i in range(num)]
    if mode == "head":
        return np.concatenate(stuff + [x], axis=0)
    else:
        return np.concatenate([x] + stuff, axis=0)


def _add_spartan_episode_images_to_replay_buffer_episode(
    *,
    observation_dict,
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    repeat_head: int,
    repeat_tail: int,
    image_shapes: Dict[str, tuple],
):
    data = {}
    # Loop through `observation_dict` and find cameras.
    camera_names_found = []
    for observation_key in observation_dict.keys():
        if observation_key not in camera_id_to_camera_name:
            continue

        camera_name = camera_id_to_camera_name[observation_key]
        camera_names_found.append(camera_name)

        if camera_name not in image_shapes:
            continue

        imgs = observation_dict[observation_key]
        imgs = _repeat(imgs, num=repeat_head, mode="head")
        # (T, H, W, 3) unnormalized.
        imgs = _repeat(imgs, num=repeat_tail, mode="tail")
        T, H, W, C = imgs.shape

        # resize
        resized = []
        h, w = image_shapes[camera_name]
        for i in range(imgs.shape[0]):
            resized.append(
                cv2.resize(imgs[i], (w, h), interpolation=cv2.INTER_LINEAR)
            )
        imgs = np.stack(resized)
        assert imgs.shape == (T, h, w, C)

        # TODO(sfeng): this is probably too restrictive. but is fine for now.
        assert imgs.dtype == np.uint8

        # We deliberately keep the imgs in datapoint as np.uint8 to keep the
        # cpu memory foot print small. Since dataloader multiplies by num_proc
        # x num_gpus, we want to save cpu memory. All preprocessing should be
        # done on the gpu side by the ml model.
        data[f"obs.{camera_name}"] = imgs

    return data


def bimanual_spartan_episode_to_replay_buffer_episode(
    *,
    observation_dict,
    action_dict,
    skill_name: str,
    lowdim_keys: List[str],
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    repeat_head: int,
    repeat_tail: int,
    image_shapes: Dict[str, tuple],
):
    action = action_dict["actions"].astype(np.float32)
    action = _repeat(action, num=repeat_head, mode="head")
    action = _repeat(action, num=repeat_tail, mode="tail")
    data = {
        "action": action,
        "skill_name": np.full((action.shape[0], 1), skill_name, dtype=object),
    }

    # loop through lowdim obs
    lowdim_obs = {}
    for name in lowdim_keys:
        if name not in observation_dict:
            continue
        obs = observation_dict[name].astype(np.float32)
        obs = _repeat(obs, num=repeat_head, mode="head")
        obs = _repeat(obs, num=repeat_tail, mode="tail")
        lowdim_obs[name] = obs

    for k, v in lowdim_obs.items():
        data[f"obs.{k}"] = v

    img_data = _add_spartan_episode_images_to_replay_buffer_episode(
        observation_dict=observation_dict,
        camera_names=camera_names,
        camera_id_to_camera_name=camera_id_to_camera_name,
        repeat_head=repeat_head,
        repeat_tail=repeat_tail,
        image_shapes=image_shapes,
    )
    data.update(img_data)

    return data


def spartan_episode_to_replay_buffer_episode(
    *,
    observation_dict,
    action_dict,
    skill_name: str,
    lowdim_keys: List[str],
    camera_names: List[str],
    camera_id_to_camera_name: Dict[str, str],
    rotation_transformer,
    repeat_head: int,
    repeat_tail: int,
    has_gripper: bool,
    image_shapes: Dict[str, tuple],
):
    """
    returns a dict with:
        obs.name: o
        action: actions
    ideall we want to return a {
        obs: {
            name: o
        },
        actions: u,
    }
    but, ReplayBuffer can only take k: nparray instead of dict.
    """
    if has_gripper:
        assert "gripper_width" in lowdim_keys
    else:
        assert "gripper_width" not in lowdim_keys

    # actions are ActionEndEffectorScripted, so we take the first 6 cols
    action = _change_pose_repr(
        action_dict["actions"][:, :6], rotation_transformer
    )
    if has_gripper:
        action = np.concatenate(
            [action, action_dict["actions"][:, 7:8].astype(np.float32)],
            axis=-1,
        )
    action = _repeat(action, num=repeat_head, mode="head")
    action = _repeat(action, num=repeat_tail, mode="tail")
    data = {
        "action": action,
        "skill_name": np.full((action.shape[0], 1), skill_name, dtype=object),
    }

    # loop through lowdim obs
    lowdim_obs = {}
    for name in lowdim_keys:
        if name == "gripper_width":
            obs = observation_dict[name].astype(np.float32)
        else:
            # assuming pose
            obs = _change_pose_repr(
                observation_dict[name], rotation_transformer
            )
        obs = _repeat(obs, num=repeat_head, mode="head")
        obs = _repeat(obs, num=repeat_tail, mode="tail")
        lowdim_obs[name] = obs

    for k, v in lowdim_obs.items():
        data[f"obs.{k}"] = v
    img_data = _add_spartan_episode_images_to_replay_buffer_episode(
        observation_dict=observation_dict,
        camera_names=camera_names,
        camera_id_to_camera_name=camera_id_to_camera_name,
        repeat_head=repeat_head,
        repeat_tail=repeat_tail,
        image_shapes=image_shapes,
    )
    data.update(img_data)

    return data


class SpartanPoseDataset(SpartanBaseDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        num_workers=8,
        max_num_episodes: Optional[int] = None,
        replay_buffer_path=None,
        is_multiarm=False,
        path_is_fully_resolved=False,
    ):
        super().__init__(
            episode_path_globs=episode_path_globs,
            shape_meta=shape_meta,
            imagenet_normalization=imagenet_normalization,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            repeat_head=repeat_head,
            repeat_tail=repeat_tail,
            stride=stride,
            has_gripper=has_gripper,
            rotation_rep=rotation_rep,
            val_ratio=val_ratio,
            seed=seed,
            num_workers=num_workers,
            max_num_episodes=max_num_episodes,
            replay_buffer_path=replay_buffer_path,
            # this doesn't support bimanual spartan data yet.
            is_multiarm=is_multiarm,
            path_is_fully_resolved=path_is_fully_resolved,
        )
        assert len(self._camera_names) == 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        to_delete = []
        to_cat = []
        for name in self._lowdim_names:
            replay_buffer_name = f"obs.{name}"
            assert replay_buffer_name in torch_data

            to_cat.append(torch_data[replay_buffer_name])
            to_delete.append(replay_buffer_name)

        for k in to_delete:
            del torch_data[k]

        # lowdim policy assumes obs is just a flat vector.
        torch_data["obs"] = torch.cat(to_cat, dim=-1)

        return torch_data

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action is always [xyz, 6d_rot, maybe_gripper]
        action_scale, action_offset, action_info = get_normalizer_params(
            self.replay_buffer["action"]
        )
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=action_scale,
            offset=action_offset,
            input_stats_dict=action_info,
        )

        if self._is_multiarm:
            # aggregate obs stats
            obs_scales = {}
            obs_offsets = {}
            obs_infos = {}
            for name in self._lowdim_names:
                x = self.replay_buffer[f"obs.{name}"]
                scale, offset, info = get_normalizer_params(x)
                obs_scales[name] = scale
                obs_offsets[name] = offset
                obs_infos[name] = info
            normalizer["obs"] = LinearNormalizer.create_manual(
                scale=obs_scales,
                offset=obs_offsets,
                input_stats_dict=obs_infos,
            )
        else:
            # aggregate obs stats
            obs_scale = []
            obs_offset = []
            obs_info = []
            for obj_idx, obj_name in enumerate(self._lowdim_names):
                x = self.replay_buffer[f"obs.{obj_name}"]
                scale, offset, info = get_normalizer_params(x)
                obs_scale.append(scale)
                obs_offset.append(offset)
                obs_info.append(info)

            obs_scale = np.concatenate(obs_scale, axis=-1)
            obs_offset = np.concatenate(obs_offset, axis=-1)
            obs_info = dict_apply_reduce(
                obs_info, lambda x: np.concatenate(x, axis=-1)
            )
            normalizer["obs"] = SingleFieldLinearNormalizer.create_manual(
                scale=obs_scale,
                offset=obs_offset,
                input_stats_dict=obs_info,
            )

        return normalizer


def _unflatten_dict_for_obs(dict_data: Dict) -> Dict:
    """
    Change the key-value structures only for the keys start with "obs." as
    follows:

       dict_data["obs.some_name"] -> dict_data["obs"]["some_name"]
    """
    obs = {}
    to_delete = []
    for k, v in dict_data.items():
        if k.startswith("obs."):
            new_key = k[4:]
            assert new_key not in obs
            obs[new_key] = v
            to_delete.append(k)
    for k in to_delete:
        del dict_data[k]
    dict_data["obs"] = obs
    return dict_data


class SpartanImageDataset(SpartanBaseDataset):
    def __init__(
        self,
        episode_path_globs: str,
        shape_meta: dict,
        imagenet_normalization: bool,
        horizon=1,
        n_obs_steps=1,
        pad_before=0,
        pad_after=0,
        repeat_head=0,
        repeat_tail=0,
        stride=1,
        has_gripper=False,
        rotation_rep="rotation_6d",
        val_ratio=0.0,
        seed=42,
        mode="np",
        compressor="blosc",
        max_num_episodes: Optional[int] = None,
        raw_rgb=True,
        num_workers=8,
        replay_buffer_path=None,
        is_multiarm=False,
        is_relative=False,
        # TODO(sfeng): should lift this into resolve_glob_type_to_list()
        path_is_fully_resolved=False,
    ):
        super().__init__(
            episode_path_globs=episode_path_globs,
            shape_meta=shape_meta,
            imagenet_normalization=imagenet_normalization,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            repeat_head=repeat_head,
            repeat_tail=repeat_tail,
            has_gripper=has_gripper,
            rotation_rep=rotation_rep,
            val_ratio=val_ratio,
            stride=stride,
            seed=seed,
            mode=mode,
            compressor=compressor,
            max_num_episodes=max_num_episodes,
            num_workers=num_workers,
            replay_buffer_path=replay_buffer_path,
            is_multiarm=is_multiarm,
            is_relative=is_relative,
            path_is_fully_resolved=path_is_fully_resolved,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if self._is_relative:
            data = change_to_relative_trajectories(
                data=data,
                # This is the time index of "T=0" or "now". The indices
                # preceding this value are in the past and those after this
                # value are in the future.
                base_index=self.n_obs_steps - 1,
                shape_meta=self._shape_meta,
            )

        # Pop skill_name from data before converting to torch because
        # skill_name is a string object and can't be converted to a tensor.
        if "skill_name" in data:
            skill_name_array = data.pop("skill_name")

        torch_data = dict_apply(data, torch.from_numpy)

        if not self._raw_rgb:
            for camera in self._camera_names:
                torch_data[f"obs.{camera}"] = (
                    torch_data[f"obs.{camera}"]
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    .float()
                    / 255.0
                )

        return _unflatten_dict_for_obs(torch_data)

    def _make_relative_proprioception_normalier_params(self):
        # TODO(sfeng): this impl iterates through the entire dataset to
        # compute the min and max for each dimension, which is pretty
        # unnecessary and won't scale to webdatasets. Should run this on
        # some large subset of our data, and have a fixed normalizer instead.
        normalizer = LinearNormalizer()
        keys = [f"obs.{k}" for k in self._lowdim_names] + ["action"]
        data_cache = {key: list() for key in keys}
        self.sampler.limit_keys(keys)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=8,
        )
        for batch in tqdm(
            dataloader, desc="iterating dataset to get normalization"
        ):
            for key in self._lowdim_names:
                data_cache[f"obs.{key}"].append(
                    copy.deepcopy(batch["obs"][key].numpy())
                )
            data_cache["action"].append(copy.deepcopy(batch["action"].numpy()))
        self.sampler.limit_keys(None)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            data_cache[key] = data_cache[key].reshape(B * T, D)

        # action
        action_normalizers = list()
        action_start_idx = {"left": 9, "right": 0}
        for arm in ["right", "left"]:
            start_idx = action_start_idx[arm]
            # pos
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx : start_idx + 3]
                    )
                )
            )
            # rot
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][:, start_idx + 3 : start_idx + 9]
                    )
                )
            )
        # gripper
        action_normalizers.append(
            get_range_normalizer_from_stat(
                array_to_stats(data_cache["action"][:, 18:])
            )
        )
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self._lowdim_names:
            stat = array_to_stats(data_cache[f"obs.{key}"])

            if key.endswith("xyz"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_6d"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("panda_hand"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer
        return normalizer

    def _make_absolute_proprioception_normalier_params(self):
        normalizer = LinearNormalizer()

        # normalizer for lowdim and actions
        stuff_to_normalize = self._lowdim_names + ["action"]
        for name in stuff_to_normalize:
            replay_buffer_name = name
            if name in self._lowdim_names:
                replay_buffer_name = f"obs.{name}"

            scale, offset, info = get_normalizer_params(
                self.replay_buffer[replay_buffer_name]
            )
            normalizer[name] = SingleFieldLinearNormalizer.create_manual(
                scale=scale,
                offset=offset,
                input_stats_dict=info,
            )

        return normalizer

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self._is_relative:
            normalizer = self._make_relative_proprioception_normalier_params()
        else:
            normalizer = self._make_absolute_proprioception_normalier_params()

        # image
        for camera_name in self._camera_names:
            normalizer[camera_name] = (
                SingleFieldLinearNormalizer.create_identity()
            )

        # if i am a feature based dataset
        assert (
            "obs.feature" not in self.replay_buffer
        ), "Shouldn't call this for feature dataset"

        return normalizer
