# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.tita_flat_config import TitaFlatCfg, TitaFlatCfgPPO
from configs.tita_rough_config import TitaRoughCfg, TitaRoughCfgPPO
from envs.no_constrains_legged_robot import Tita

from global_config import ROOT_DIR, ENVS_DIR
import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry
import torch


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    for i in range(int(8*env.max_episode_length)):
        actions = 0.*torch.ones(env.num_envs, env.num_actions, device=env.device)
        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
    print("Done")

if __name__ == '__main__':
    task_registry.register("tita_constraint",LeggedRobot,TitaConstraintRoughCfg(),TitaConstraintRoughCfgPPO())
    args = get_args()
    test_env(args)
