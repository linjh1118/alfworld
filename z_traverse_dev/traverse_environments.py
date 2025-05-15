import os
import json
from pathlib import Path
import subprocess
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

from alfworld.agents.environment.alfred_tw_env import TASK_TYPES

def init_xvfb(display_id=100):
    _xvfb_proc = subprocess.Popen(
        ["Xvfb", f":{display_id}", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{display_id}"

def traverse_alfworld_environments():
    # Load the base configuration
    config = generic.load_config()
    
    # Choose environment type ('AlfredTWEnv' for text-world or 'AlfredThorEnv' for embodied environment)
    # env_type = 'AlfredTWEnv'
    env_type = 'AlfredThorEnv'
    
    # Initialize training environment to access all games
    alfred_env = get_environment(env_type)(config, train_eval="train")
    
    # Print total number of games
    print(f"Total number of training games: {alfred_env.num_games}")
    
    # Print task types
    task_types = {}
    for tt_id in config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_type_name = TASK_TYPES[tt_id]
            task_types[tt_id] = task_type_name
            print(f"Task Type {tt_id}: {task_type_name}")
    
    # Initialize environment with batch size 1 (one game at a time)
    env = alfred_env.init_env(batch_size=1)
    
    
    
    # Counts per task type
    task_type_counts = {tt_id: 0 for tt_id in task_types}
    
    # Iterate through all games
    # for i in range(alfred_env.num_games):
    for i in range(5):
        # Set random seed for reproducibility
        np.random.seed(i)
        env.seed(i)
        
        # Reset environment
        obs, infos = env.reset()
        
        # Get game file path
        game_file = infos["extra.gamefile"][0]
        
        # Get the corresponding traj_data.json file which has task info
        # AlfredTWEnv: game file is at "path/to/game.tw-pddl"
        # traj_data.json is in the same directory
        game_dir = os.path.dirname(game_file)
        traj_json_path = os.path.join(game_dir, "traj_data.json")
        # traj_json_path 位于 game_dir / xxx / traj_data.json，用path找出这种pattern的文件，选择最后一个
        traj_json_path = list(Path(game_dir).glob("*/traj_data.json"))[-1]
        
        
        # Load traj_data to get task information
        with open(traj_json_path, 'r') as f:
            traj_data = json.load(f)
            
        # Get task type
        task_type_id = None
        for tid in task_types:
            if task_types[tid] == traj_data['task_type']:
                task_type_id = tid
                task_type_counts[tid] += 1
                break
        
        # Print current game info
        print(f"Game {i+1}/{alfred_env.num_games}:")
        print(f"  File: {game_file}")
        print(f"  Task Type: {traj_data['task_type']} (ID: {task_type_id})")
        if 'scene' in traj_data:
            print(f"  Scene: {traj_data['scene']['scene_num']}")
        print(f"  Task Description: {traj_data['turk_annotations']['anns'][0]['task_desc']}")
        print("")
    
    # Print summary
    print("\nTask Type Summary:")
    for task_id, count in task_type_counts.items():
        print(f"  {task_types[task_id]} (Type {task_id}): {count} games")
    print(f"Total Games: {alfred_env.num_games}")

if __name__ == "__main__":
    init_xvfb()
    traverse_alfworld_environments()