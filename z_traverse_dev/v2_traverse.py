import os
import json
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES

def init_xvfb(display_id=100):
    """Initialize virtual display for visualization"""
    _xvfb_proc = subprocess.Popen(
        ["Xvfb", f":{display_id}", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{display_id}"
    return _xvfb_proc

def save_visualization(frames, output_dir, env_idx):
    """保存环境截图到输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果frames是单帧图像
    if len(frames.shape) == 3:
        plt.figure(figsize=(10, 8))
        plt.imshow(frames)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"env_{env_idx}.png"))
        plt.close()
    # 如果frames是多帧图像
    elif len(frames.shape) == 4:
        for i, frame in enumerate(frames):
            plt.figure(figsize=(10, 8))
            plt.imshow(frame)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"env_{env_idx}_frame_{i}.png"))
            plt.close()

def main():
    # 初始化虚拟显示
    xvfb_proc = init_xvfb()
    
    try:
        # 加载配置
        config = generic.load_config()
        
        # 创建输出目录
        output_dir = "alfworld_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置环境类型 - 使用Thor环境以便获取可视化
        env_type = 'AlfredThorEnv'  # 'AlfredTWEnv' for text-world
        
        # 初始化训练环境以访问所有游戏
        alfred_env = get_environment(env_type)(config, train_eval="train")
        
        # 打印任务类型
        task_types = {}
        for tt_id in config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_type_name = TASK_TYPES[tt_id]
                task_types[tt_id] = task_type_name
                print(f"Task Type {tt_id}: {task_type_name}")
        
        # 初始化环境（batch_size=1，一次处理一个游戏）
        env = alfred_env.init_env(batch_size=1)
        
        # 获取所有游戏文件
        game_files = alfred_env.json_file_list
        print(f"Total number of training games: {len(game_files)}")
        
        # 创建一个日志文件记录环境信息
        log_file = os.path.join(output_dir, "environments_log.txt")
        
        # 限制处理的游戏数量，避免处理时间过长
        max_games = min(20, len(game_files))  # 最多处理20个游戏
        
        # 创建可视化摘要文件
        with open(log_file, 'w') as f:
            f.write(f"ALFWorld Environments Visualization\n")
            f.write(f"Total games: {len(game_files)}\n")
            f.write(f"Visualizing {max_games} games\n\n")
            
            # 遍历游戏文件
            for i, game_file in enumerate(tqdm(game_files[:max_games])):
                # 重置环境
                obs, infos = env.reset()
                
                # 获取游戏信息
                game_path = infos["extra.gamefile"][0]
                
                # 查找对应的traj_data.json文件
                game_dir = os.path.dirname(game_path)
                traj_json_paths = list(Path(game_dir).glob("*/traj_data.json"))
                if not traj_json_paths:
                    traj_json_path = os.path.join(game_dir, "traj_data.json")
                else:
                    traj_json_path = traj_json_paths[-1]
                
                # 加载任务数据
                with open(traj_json_path, 'r') as traj_file:
                    traj_data = json.load(traj_file)
                
                # 获取任务类型
                task_type = traj_data['task_type']
                task_type_id = None
                for tid in task_types:
                    if task_types[tid] == task_type:
                        task_type_id = tid
                        break
                
                # 记录游戏信息
                f.write(f"Game {i+1}/{max_games}:\n")
                f.write(f"  File: {game_file}\n")
                f.write(f"  Task Type: {task_type} (ID: {task_type_id})\n")
                if 'scene' in traj_data:
                    f.write(f"  Scene: {traj_data['scene']['scene_num']}\n")
                f.write(f"  Task Description: {traj_data['turk_annotations']['anns'][0]['task_desc']}\n")
                f.write(f"  Initial Observation: {obs[0]}\n\n")
                
                # 获取并保存环境图像
                try:
                    frames = env.get_frames()
                    save_visualization(frames, os.path.join(output_dir, "images"), i)
                    
                    # 尝试获取探索帧（如果可用）
                    try:
                        exploration_frames = env.get_exploration_frames()
                        if exploration_frames and len(exploration_frames[0]) > 0:
                            for j, exp_frame in enumerate(exploration_frames[0]):
                                plt.figure(figsize=(10, 8))
                                plt.imshow(exp_frame)
                                plt.axis('off')
                                plt.savefig(os.path.join(output_dir, "exploration", f"env_{i}_explore_{j}.png"))
                                plt.close()
                    except Exception as e:
                        print(f"Could not get exploration frames: {e}")
                        
                except Exception as e:
                    print(f"Error capturing frames for game {i}: {e}")
                
                # 执行一个简单的操作（如"look"）以获取更多信息
                try:
                    obs, _, dones, infos = env.step(["look"])
                    f.write(f"  After 'look' action: {obs[0]}\n\n")
                    
                    # 保存执行动作后的帧
                    frames = env.get_frames()
                    save_visualization(frames, os.path.join(output_dir, "after_look"), i)
                except Exception as e:
                    print(f"Error executing 'look' action: {e}")
                
                # 添加分隔线
                f.write("-" * 80 + "\n\n")
        
        print(f"Visualization completed. Results saved to {output_dir}")
        
    finally:
        # 关闭环境和虚拟显示
        try:
            env.close()
        except:
            pass
        
        try:
            xvfb_proc.terminate()
        except:
            pass

if __name__ == "__main__":
    main()