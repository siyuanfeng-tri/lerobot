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
"""
Contains utilities to process raw data format of npz files from TRI sim environments.
"""

import json
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Sequence, Value

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)

DATA_FILE = "filtered_data.json"


def check_format(raw_dir: Path) -> bool:
    # only frames from simulation are uncompressed
    required_keys = ("x", "y", "vx", "vy", "fx", "fy")

    assert (raw_dir / DATA_FILE).exists(), f"Required file {DATA_FILE} not found in {raw_dir}"
    # Now load the file and check if it is in the correct format
    with open(raw_dir / DATA_FILE, "r") as f:
        data = json.load(f)

    for k, v in data.items():
        assert isinstance(k, str), f"Key {k} is not a string."
        assert isinstance(v, list), f"Value {v} is not a list."

        for i, frame in enumerate(v):
            assert isinstance(frame, dict), f"Frame {i} is not a dict."

            for k2, v2 in frame.items():
                assert isinstance(k2, str), f"Key {k2} is not a string."
                assert isinstance(v2, (int, float)), f"Value {v2} is not a string, int, float or list."
            assert all(k2 in frame for k2 in required_keys), f"Frame {i} is missing required keys."

    return True


def load_from_raw(
    raw_dir: Path, videos_dir: Path, fps: int = 30, video: bool = True, episodes: list[int] | None = None
):
    with open(raw_dir / DATA_FILE, "r") as f:
        raw_data = json.load(f)
    num_episodes = len(raw_data)
    ep_id_to_ep_key = list(raw_data.keys())

    # Only video form is supported for now
    assert not video, "There are no image data in this environment."

    state_variables = [
        "x",
        "y",
        "vx",
        "vy",
    ]
    action_variables = ["fx", "fy"]

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_key = ep_id_to_ep_key[ep_idx]
        ep_data = raw_data[ep_key]

        actions = np.array([[frame[var] for var in action_variables] for frame in ep_data])
        num_frames = actions.shape[0]

        ep_dict = {}
        observations = np.array([[frame[var] for var in state_variables] for frame in ep_data])

        ep_dict["action"] = torch.from_numpy(actions)
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["observation.state"] = torch.from_numpy(observations)
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True
        ep_dict["next.done"] = done
        assert isinstance(ep_idx, int)
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video: bool) -> Dataset:
    features = {}

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = False,
    episodes: list[int] | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
