import os
import json
import random
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES
import pandas as pd
from pathlib import Path

from nlp_tools import ImageUtils

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
    # Create DataFrame from collected data
    if df is None:
        df = pd.DataFrame(data_list)
    
    # Define CSS styles for the HTML output
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
    
    # Convert DataFrame to HTML
    df_html = df.to_html(render_links=True, escape=False)
    
    # Combine everything into final HTML
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
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def save_visualization(frame, output_dir, env_idx):
    """Save environment screenshot and return relative path"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    image_path = os.path.join(output_dir, f"env_{env_idx}.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(frame)
    plt.axis('off')
    plt.savefig(image_path)
    plt.close()
    
    image_path = os.path.abspath(image_path)    
    return ImageUtils.image_to_show(image_path)
    

def main():
    # 初始化虚拟显示
    xvfb_proc = init_xvfb()
    
    try:
        # 加载配置
        config = generic.load_config()
        
        # 创建输出目录
        output_dir = "alfworld_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置环境类型
        env_type = 'AlfredThorEnv'
        
        # 初始化训练环境
        alfred_env = get_environment(env_type)(config, train_eval="train")
        env = alfred_env.init_env(batch_size=1)
        
        # 获取游戏文件
        game_files = alfred_env.json_file_list
        print(f"Total number of training games: {len(game_files)}")
        random.shuffle(game_files)
        
        # 限制处理的游戏数量
        # max_games = min(20, len(game_files))
        max_games = 100
        
        # 收集环境数据
        environment_data = []
        
        # 遍历游戏文件
        for i, game_file in enumerate(tqdm(game_files[:max_games])):
            env_data = {}
            
            # 重置环境
            obs, infos = env.reset()
            
            # 获取游戏信息
            game_path = infos["extra.gamefile"][0]
            traj_json_path = os.path.join(game_path, "traj_data.json")
            
            # 加载任务数据
            with open(traj_json_path, 'r') as traj_file:
                traj_data = json.load(traj_file)
            
            # 收集环境信息
            # env_data['Environment ID'] = i + 1
            env_data['Task Type'] = traj_data['task_type']
            env_data['Scene'] = f"Scene {traj_data['scene']['scene_num']}" if 'scene' in traj_data else 'N/A'
            env_data['Task Description'] = traj_data['turk_annotations']['anns'][0]['task_desc']
            env_data['Initial Observation'] = obs[0].replace('\n', '<br>')
            
            # 获取并保存环境图像
            try:
                frames = env.get_frames()
                if len(frames.shape) == 3:  # 单帧
                    env_data['Environment View'] = save_visualization(frames, os.path.join(output_dir, "images"), i)
                elif len(frames.shape) == 4:  # 多帧
                    env_data['Environment View'] = save_visualization(frames[0], os.path.join(output_dir, "images"), i)
            except Exception as e:
                env_data['Environment View'] = f"Error capturing frame: {str(e)}"
            
            environment_data.append(env_data)
        
        
        
        # 生成HTML报告
        html_path = os.path.join(output_dir, "environment_report.html")
        df = pd.DataFrame(environment_data)
        # group by task type
        df = df.sort_values('Task Type')
        generate_html_report(df=df, output_path=html_path)
        
        print(f"Visualization completed. HTML report saved to {html_path}")
        
    finally:
        # 清理资源
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
    

# 改成同时启动8个显示器，从而来提高效率。
# 注意每一个进程里，只能有一个显示器。
