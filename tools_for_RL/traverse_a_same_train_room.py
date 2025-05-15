import os
import json
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

from nlp_tools import ImageUtils

class FixedGameAlfredThorEnv(AlfredThorEnv):
    """
    修改版的AlfredThorEnv，允许指定固定的游戏文件
    """
    
    def __init__(self, config, train_eval="train", fixed_game_file=None):
        super().__init__(config, train_eval)
        self.fixed_game_file = fixed_game_file
        
        # 如果指定了固定游戏文件，则覆盖json_file_list
        if self.fixed_game_file:
            self.json_file_list = [self.fixed_game_file]
    
    def reset(self):
        # 设置任务
        batch_size = self.batch_size
        
        # 如果指定了固定游戏文件，则所有环境都使用同一个游戏文件
        if self.fixed_game_file:
            tasks = [self.fixed_game_file] * batch_size
        else:
            # 否则使用原来的随机选择逻辑
            if self.train_eval == 'train':
                tasks = random.sample(self.json_file_list, k=batch_size)
            else:
                if len(self.json_file_list)-batch_size > batch_size:
                    tasks = [self.json_file_list.pop(random.randrange(len(self.json_file_list))) for _ in range(batch_size)]
                else:
                    tasks = random.sample(self.json_file_list, k=batch_size)
                    self.get_env_paths()
        
        # 重置所有环境
        for n in range(batch_size):
            self.action_queues[n].put((None, True, tasks[n]))
        
        obs, dones, infos = self.wait_and_get_info()
        return obs, infos

def init_xvfb(display_id=100):
    """Initialize virtual display for visualization"""
    _xvfb_proc = subprocess.Popen(
        ["Xvfb", f":{display_id}", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{display_id}"
    return _xvfb_proc

def generate_html_report(data_list=None, df=None, output_path=None):
    """Generate HTML report from environment data"""
    if df is None:
        df = pd.DataFrame(data_list)
    
    css_styles = """<style>
        th, td {
            max-width: 800px;
            word-wrap: break-word;
        }
        th {
            background-color: #fff;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        th:before {
            content: "";
            position: absolute;
            top: -1px;
            bottom: -1px;
            left: -1px;
            right: -1px;
            border: 1px solid LightGrey;
            z-index: -1;
        }
        img {
            max-width: 500px;
            height: auto;
        }
    </style>
    """
    
    df_html = df.to_html(render_links=True, escape=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ALFWorld Environment Visualization</title>
        {css_styles}
    </head>
    <body>
        <h1>ALFWorld Environment Visualization</h1>
        {df_html}
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def save_visualization(frame, output_dir, env_idx):
    """Save environment screenshot and return relative path"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_path = os.path.join(output_dir, f"env_{env_idx}.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.axis('off')
    plt.savefig(image_path)
    plt.close()
    
    image_path = os.path.abspath(image_path)    
    return ImageUtils.image_to_show(image_path)

def process_env_instance(args):
    """Process a single environment instance with its own display"""
    env_idx, config, game_file, output_dir = args
    display_id = 100 + env_idx  # Unique display ID for each process
    
    # Initialize virtual display for this process
    xvfb_proc = init_xvfb(display_id)
    
    try:
        # 创建自定义环境
        alfred_env = FixedGameAlfredThorEnv(config, train_eval="train", fixed_game_file=game_file)
        env = alfred_env.init_env(batch_size=1)
        
        # 重置环境
        obs, infos = env.reset()
        
        # 获取游戏信息
        game_path = infos["extra.gamefile"][0]
        traj_json_path = os.path.join(game_path, "traj_data.json")
        
        # 加载任务数据
        with open(traj_json_path, 'r') as traj_file:
            traj_data = json.load(traj_file)
        
        # 收集环境信息
        env_data = {}
        env_data['Environment ID'] = env_idx + 1
        env_data['Task Type'] = traj_data['task_type']
        env_data['Scene'] = f"Scene {traj_data['scene']['scene_num']}" if 'scene' in traj_data else 'N/A'
        env_data['Task Description'] = traj_data['turk_annotations']['anns'][0]['task_desc']
        env_data['Initial Observation'] = obs[0].replace('\n', '<br>')
        env_data['Game File'] = game_file
        
        # 获取并保存环境图像
        try:
            frames = env.get_frames()
            if len(frames.shape) == 3:  # 单帧
                env_data['Environment View'] = save_visualization(frames, os.path.join(output_dir, "images"), env_idx)
            elif len(frames.shape) == 4:  # 多帧
                env_data['Environment View'] = save_visualization(frames[0], os.path.join(output_dir, "images"), env_idx)
        except Exception as e:
            env_data['Environment View'] = f"Error capturing frame: {str(e)}"
        
        return env_data
    finally:
        # 清理
        try:
            env.close()
        except:
            pass
        try:
            xvfb_proc.terminate()
        except:
            pass

def main():
    # 加载配置
    config = generic.load_config()
    
    # 创建输出目录
    output_dir = "alfworld_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # 初始化环境以获取游戏文件
    env_type = 'AlfredThorEnv'
    alfred_env = get_environment(env_type)(config, train_eval="train")
    game_files = alfred_env.json_file_list
    alfred_env.close()  # 关闭初始环境
    
    print(f"Total number of training games: {len(game_files)}")
    
    # 选择一个特定的游戏文件
    specific_game_file = game_files[0]  # 你可以选择任何一个游戏文件或通过命令行参数指定
    print(f"Selected game file: {specific_game_file}")
    
    # 准备多进程参数
    num_instances = 8  # 创建8个相同环境的实例
    args_list = [(i, config, specific_game_file, output_dir) for i in range(num_instances)]
    
    # 并行处理环境实例
    num_processes = min(num_instances, cpu_count())
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_env_instance, args_list), total=len(args_list)))
    
    # 生成HTML报告
    html_path = os.path.join(output_dir, "environment_report.html")
    df = pd.DataFrame(results)
    generate_html_report(df=df, output_path=html_path)
    
    print(f"Visualization completed. HTML report saved to {html_path}")
    print(f"All {num_instances} environments were initialized with the same game file: {specific_game_file}")

if __name__ == "__main__":
    main()