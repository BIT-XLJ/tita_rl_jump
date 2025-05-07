import os
import numpy as np

import math
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch
import torch
# 初始化Gym
gym = gymapi.acquire_gym()

# 创建模拟环境
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0
sim_params.physx.contact_offset = 0.02

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 创建地面平面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 设置摄像机
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
cam_pos = gymapi.Vec3(3, 2, 1)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 加载URDF机器人
asset_root = "/home/djh/tita/tita_rl/resources/tita/urdf"  # 替换为你的URDF文件所在目录
asset_file = "tita_description.urdf"  # 替换为你的URDF文件名

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # 固定根关节/基座
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True
asset_options.disable_gravity = False

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 获取关节信息
num_dofs = gym.get_asset_dof_count(robot_asset)
dof_props = gym.get_asset_dof_properties(robot_asset)

# 打印关节信息
print(f"Number of DOFs: {num_dofs}")
print("Joint properties:")
for i in range(num_dofs):
    joint_name = gym.get_asset_dof_name(robot_asset, i)
    print(f"  Joint {i} ({joint_name}):")
    print(f"    Lower limit: {dof_props['lower'][i]}")
    print(f"    Upper limit: {dof_props['upper'][i]}")

# 设置正弦波参数
# frequencies = np.linspace(0.1, 0.5, num_dofs)  # 每个关节不同的频率
# amplitudes = (dof_props['upper'] - dof_props['lower']) * 0.3  # 使用30%的运动范围
# offsets = (dof_props['upper'] + dof_props['lower']) * 0.5  # 中位值

# 创建环境
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# 创建actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.0)
pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.0)

robot_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)

# 准备状态Tensor
_gym = gym
_gym.prepare_sim(sim)
root_tensor = _gym.acquire_actor_root_state_tensor(sim)
dof_state_tensor = _gym.acquire_dof_state_tensor(sim)
root_tensor = gymtorch.wrap_tensor(root_tensor)
dof_state_tensor = gymtorch.wrap_tensor(dof_state_tensor)

# 主循环
frame_count = 0
cycle_time = 0.6
while not gym.query_viewer_has_closed(viewer):
    # 更新时间
    frame_count += 1
    t = frame_count * sim_params.dt
    
    # 为每个关节生成正弦波位置
    new_dof_pos = np.zeros(num_dofs, dtype=np.float32)
    new_dof_pos[1] = 1.1
    new_dof_pos[2] = -2.1
    new_dof_pos[5] = 1.1
    new_dof_pos[6] = -2.1

    phase =  torch.tensor(t / cycle_time - torch.floor(torch.tensor(t / cycle_time)))
    sin_pos = torch.sin(2 * torch.pi * phase)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()
    ref_dof_pos = np.zeros(num_dofs, dtype=np.float32)
    scale_1 = 0.3  #0.7
    scale_2 = -2 * scale_1
    # left foot stance phase set to default joint pos
    if phase>0.25:
        sin_pos_l = 0
    ref_dof_pos[1] = sin_pos_l * scale_1
    ref_dof_pos[2] = sin_pos_l * scale_2
    # right foot stance phase set to default joint pos
    if phase>0.25:
        sin_pos_r = 0
    ref_dof_pos[5] = sin_pos_r * scale_1
    ref_dof_pos[6] = sin_pos_r * scale_2
    # ref_dof_pos[1] = sin_pos_r * scale_1
    # ref_dof_pos[2] = sin_pos_r * scale_2
    # Double support phase
    # ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

    new_dof_pos -= ref_dof_pos

    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # 填充位置和速度
    for i in range(num_dofs):
        dof_states[i]['pos'] = new_dof_pos[i]  # 设置位置
        dof_states[i]['vel'] = 0.0  # 设置速度
    # 设置关节位置

    gym.set_actor_dof_states(env, robot_handle, dof_states, gymapi.STATE_POS)
    
    # 更新模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 更新查看器
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # 同步帧
    gym.sync_frame_time(sim)

# 清理
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)