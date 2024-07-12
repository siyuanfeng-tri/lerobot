#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from pathlib import Path
from typing import Callable

import datasets
import torch
import torch.utils

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    load_episode_data_index,
    load_hf_dataset,
    load_info,
    load_previous_and_future_frames,
    load_stats,
    load_videos,
    reset_episode_index,
)
from lerobot.common.datasets.video_utils import VideoFrame, load_from_videos
import numpy as np

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
CODEBASE_VERSION = "v1.4"


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.cpu().detach().numpy()


def mat_to_pose9d(mat):
    """
    Takes a [N, 4, 4] np array of transformations and convert to [N, 9].
    Inverse of pose9d_to_mat()
    """
    # TODO(sfeng): test this function
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    d6 = mat_to_rot6d(rotmat)
    d9 = np.concatenate([pos, d6], axis=-1)
    return d9


def pose9d_to_mat(d9):
    """
    Takes a [N, 9] np array and convert to [N, 4, 4] transformations.
    Each row of input is assumed to be xyz, 6d rotation.
    """
    # TODO(sfeng): test this function
    pos = d9[..., :3]
    d6 = d9[..., 3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    out[..., :3, :3] = rotmat
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1
    return out


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out


def rot6d_to_mat(d6):
    """
    Takes a [N, 6] np array and convert to [N, 3, 3] rotation matrices.
    The input is assumed to be the first 2 rows of a rotation matrix.
    """
    # TODO(sfeng): test this function
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out


def mat_to_rot6d(mat):
    """
    Takes a [N, 3, 3] np array and convert to [N, 6] array.
    Inverse of rot6d_to_mat().
    """
    # TODO(sfeng): test this function
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def convert_pose_mat_rep(pose_mat, X_A_C, pose_rep="abs", backward=False):
    """
    Convert the [N, 4, 4] `pose_mat` pose trajectory depending on `pose_rep`,
    which can be "abs" or "relative".
    In "abs" mode:
        - `pose_mat` is directly returned.
    In "relative" mode:
        - when "backward" is False, `pose_mat` is treated as X_A_B_traj,
            and this func returns X_C_B_traj.
        - when "backward" is True, `pose_mat` is treated as X_C_B_traj,
            and this func returns X_A_B_traj.

    A concrete example for converting a trajectory of ee pose in world frame
    (X_W_EE) to a trajectory of ee pose wrt some other frame in world (X_W_F)
    X_F_EE = convert_pose_mat_rep(
        X_W_EE, X_W_F, pose_rep="relative", backward=False
    )
    To convert X_F_EE to X_W_EE again:
    X_W_EE = convert_pose_mat_rep(
        X_F_EE, X_W_F, pose_rep="relative", backward=True
    )
    """
    if not backward:
        # training transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "relative":
            out = np.linalg.inv(X_A_C) @ pose_mat
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")

    else:
        # eval transform
        if pose_rep == "abs":
            return pose_mat
        elif pose_rep == "relative":
            out = X_A_C @ pose_mat
            return out
        else:
            raise RuntimeError(f"Unsupported pose_rep: {pose_rep}")


def change_to_relative_batch(item, base_index):
    old_state = item["observation.state"].clone()
    old_action = item["action"].clone()
    # item['observation.state'] is 2 x 20
    # item['action'] is 16 x 20

    # state is this order
    #"robot__actual__poses__right::panda__xyz",
    #"robot__actual__poses__right::panda__rot_6d",
    #"robot__actual__poses__left::panda__xyz",
    #"robot__actual__poses__left::panda__rot_6d",
    #"robot__actual__grippers__right::panda_hand",
    #"robot__actual__grippers__left::panda_hand",
    pose_start_idx = {
        "left": 9,
        "right": 0,
    }
    X_W_arm_traj = {}

    def _get_pose_traj(vec, start_idx):
        return pose9d_to_mat(
            _to_numpy(vec[:, start_idx:start_idx + 9])
        )

    for arm in ["left", "right"]:
        start_idx = pose_start_idx[arm]
        X_W_arm_traj[arm] = _get_pose_traj(item['observation.state'], start_idx)

    vec_other_arm_trajs = []
    # add relative between left and right
    new_state = item["observation.state"].clone()
    for arm in ["left", "right"]:
        other_arm = "right" if arm == "left" else "left"
        X_W_other_traj = _get_pose_traj(
            item['observation.state'], pose_start_idx[other_arm]
        )

        X_other_arm_traj = convert_pose_mat_rep(
            X_W_arm_traj[arm],
            X_A_C=X_W_other_traj[base_index],
            pose_rep="relative",
            backward=False,
        )
        vec_other_arm_trajs.append(
            torch.from_numpy(mat_to_pose9d(X_other_arm_traj))
        )

    # action is this order
    # r xyz, r rot6d, r gripper
    # l xyz, l rot6d, l gripper
    action_start_idx = {"left": 10, "right": 0}

    new_action = item["action"].clone()
    for arm in ["left", "right"]:
        # change base frame to current time steps' arm pose
        X_cur_arm_traj = convert_pose_mat_rep(
            X_W_arm_traj[arm],
            X_A_C=X_W_arm_traj[arm][base_index],
            pose_rep="relative",
            backward=False,
        )
        vec_cur_arm_traj = mat_to_pose9d(X_cur_arm_traj)
        # overwrite obs
        pose_start = pose_start_idx[arm]
        pose_end = pose_start + 9
        new_state[:, pose_start:pose_end] = torch.from_numpy(vec_cur_arm_traj)

        if "action" in item:
            u_start = action_start_idx[arm]
            u_end = u_start + 9
            X_W_action_traj = pose9d_to_mat(
                _to_numpy(item["action"][:, u_start:u_end])
            )
            # change base frame to current time steps' arm pose
            X_cur_action_traj = convert_pose_mat_rep(
                X_W_action_traj,
                X_A_C=X_W_arm_traj[arm][base_index],
                pose_rep="relative",
                backward=False,
            )

            # overwrite action
            vec_cur_action_traj = mat_to_pose9d(X_cur_action_traj)
            new_action[:, u_start:u_end] = torch.from_numpy(vec_cur_action_traj)

    # append interhand delta pose
    new_state = torch.cat([new_state] + vec_other_arm_trajs, axis=-1)

    item['observation.state'] = new_state
    item['action'] = new_action
    return item, old_state, old_action


def change_rel_action_to_abs(
    *,
    X_W_arm,
    vec_cur_action_traj,
):
    """
    Converts relative action trajectory back to world frame using current end
    effector poses in the world frame.
    Args:
        X_W_arm: dict from arm name to 4d transformation matrix. Keys should
            be [left::panda, right::panda]
        vec_cur_action_traj: (N, 20) actions. The layout need to match spartan
            format
    """

    # go from relative tback to absolute
    action_start_idx = {"left": 10, "right": 0}
    vec_W_action_traj = _to_numpy(vec_cur_action_traj)

    for arm in ["left", "right"]:
        start_idx = action_start_idx[arm]
        X_cur_action_traj = pose9d_to_mat(
            _to_numpy(vec_cur_action_traj[:, start_idx : start_idx + 9])
        )
        X_W_action_traj = convert_pose_mat_rep(
            X_cur_action_traj,
            X_A_C=X_W_arm[arm],
            pose_rep="relative",
            backward=True,
        )
        my_vec_W_action_traj = mat_to_pose9d(X_W_action_traj)
        vec_W_action_traj[:, start_idx : start_idx + 9] = my_vec_W_action_traj

    return vec_W_action_traj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
        is_relative_traj: bool = False,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.version = version
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # load data from hub or locally when root is provided
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        self.hf_dataset = load_hf_dataset(repo_id, version, root, split)
        if split == "train":
            self.episode_data_index = load_episode_data_index(repo_id, version, root)
        else:
            self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
            self.hf_dataset = reset_episode_index(self.hf_dataset)
        self.stats = load_stats(repo_id, version, root)
        self.info = load_info(repo_id, version, root)
        if self.video:
            self.videos_dir = load_videos(repo_id, version, root)
            self.video_backend = video_backend if video_backend is not None else "pyav"
        self.is_relative_traj = is_relative_traj

        assert self.is_relative_traj

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.
        Returns False if it only loads images from png files.
        """
        return self.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset.unique("episode_index"))

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        if self.is_relative_traj:
            cur_idx = 1
            item, old_state, old_action = change_to_relative_batch(item, base_index = cur_idx)

            X_W_arm = {}
            pose_start_idx = {
                "left": 9,
                "right": 0,
            }
            for arm in ["left", "right"]:
                start_idx = pose_start_idx[arm]
                X_W_arm[arm] = pose9d_to_mat(
                    _to_numpy(old_state[cur_idx, start_idx:start_idx+9])
                )

            recovered_old_action = change_rel_action_to_abs(
                X_W_arm=X_W_arm,
                vec_cur_action_traj=item['action'],
            )

        if self.video:
            item = load_from_videos(
                item,
                self.video_frame_keys,
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Version: '{self.version}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str = "from_preloaded",
        version: str | None = CODEBASE_VERSION,
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
        # additional preloaded attributes
        hf_dataset=None,
        episode_data_index=None,
        stats=None,
        info=None,
        videos_dir=None,
        video_backend=None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from existing data and attributes instead of loading from the filesystem.

        It is especially useful when converting raw data into LeRobotDataset before saving the dataset
        on the filesystem or uploading to the hub.

        Note: Meta-data attributes like `repo_id`, `version`, `root`, etc are optional and potentially
        meaningless depending on the downstream usage of the return dataset.
        """
        # create an empty object of type LeRobotDataset
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.version = version
        obj.root = root
        obj.split = split
        obj.image_transforms = transform
        obj.delta_timestamps = delta_timestamps
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = episode_data_index
        obj.stats = stats
        obj.info = info if info is not None else {}
        obj.videos_dir = videos_dir
        obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                version=version,
                root=root,
                split=split,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]
        # Check that some properties are consistent across datasets. Note: We may relax some of these
        # consistency requirements in future iterations of this class.
        for repo_id, dataset in zip(self.repo_ids, self._datasets, strict=True):
            if dataset.info != self._datasets[0].info:
                raise ValueError(
                    f"Detected a mismatch in dataset info between {self.repo_ids[0]} and {repo_id}. This is "
                    "not yet supported."
                )
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_data_keys = set()
        intersection_data_keys = set(self._datasets[0].hf_dataset.features)
        for dataset in self._datasets:
            intersection_data_keys.intersection_update(dataset.hf_dataset.features)
        if len(intersection_data_keys) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. The "
                "multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, dataset in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(dataset.hf_dataset.features).difference(intersection_data_keys)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_data_keys.update(extra_keys)

        self.version = version
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats(self._datasets)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.features.items() if k not in self.disabled_data_keys})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_samples for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_samples:
                start_idx += dataset.num_samples
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_data_keys:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Version: '{self.version}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
