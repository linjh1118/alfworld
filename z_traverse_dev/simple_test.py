import subprocess
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import json
import os
import random
from tqdm import tqdm
import argparse

def evaluate_random_agent(config, env_type, num_games=50, max_steps_per_episode=50):
    """
    评估随机agent在ALFWorld环境中的表现
    
    Args:
        config: 配置字典
        env_type: 环境类型 ('AlfredTWEnv' 或 'AlfredThorEnv')
        num_games: 测试游戏数量
        max_steps_per_episode: 每个episode的最大步数
    
    Returns:
        评估结果字典
    """
    # 设置环境
    config['env']['type'] = env_type
    env = get_environment(env_type)(config, train_eval='eval_out_of_distribution')
    env = env.init_env(batch_size=1)
    
    success_list = []
    gc_success_list = []
    steps_list = []
    game_info_list = []
    
    # 测试多个游戏
    for _ in tqdm(range(num_games), desc=f"评估随机Agent ({env_type})"):
        obs, info = env.reset()
        done = False
        steps = 0
        
        # 在单个游戏中循环
        while not done and steps < max_steps_per_episode:
            # 随机选择动作
            if env_type == 'AlfredTWEnv':
                # 对于文本环境，从有效动作中随机选择
                admissible_commands = list(info['admissible_commands'])
                if len(admissible_commands[0]) > 0:
                    action = [random.choice(admissible_commands[0])]
                else:
                    action = ["look"]
            else:
                # 对于THOR环境，从一组基本动作中随机选择
                basic_actions = ["look", "move ahead", "turn left", "turn right", 
                               "pick up", "put", "open", "close", "toggle", "inventory"]
                action = [random.choice(basic_actions)]
            
            # 执行动作
            obs, _, dones, infos = env.step(action)
            done = dones[0]
            steps += 1
        
        # 记录结果
        success = float(infos['won'][0])
        gc_success = float(infos['goal_condition_success_rate'][0]) if 'goal_condition_success_rate' in infos else 0.0
        
        success_list.append(success)
        gc_success_list.append(gc_success)
        steps_list.append(steps)
        
        game_info = f"{infos['extra.gamefile'][0]}, score: {success}, goal_condition_score: {gc_success}, steps: {steps}"
        game_info_list.append(game_info)
    
    # 汇总结果
    avg_success = np.mean(success_list)
    avg_gc_success = np.mean(gc_success_list) 
    avg_steps = np.mean(steps_list)
    
    print(f"==== 随机Agent评估结果 ({env_type}) ====")
    print(f"平均成功率: {avg_success:.4f}")
    print(f"平均目标条件成功率: {avg_gc_success:.4f}")
    print(f"平均步数: {avg_steps:.2f}")
    
    results = {
        'env_type': env_type,
        'num_games': num_games,
        'max_steps': max_steps_per_episode,
        'average_success': float(avg_success),
        'average_goal_condition_success': float(avg_gc_success),
        'average_steps': float(avg_steps),
        'success_list': success_list,
        'gc_success_list': gc_success_list,
        'steps_list': steps_list,
        'game_info': game_info_list
    }
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval_config.yaml', help='配置文件路径')
    parser.add_argument('--env_type', default='AlfredThorEnv', choices=['AlfredTWEnv', 'AlfredThorEnv'], help='环境类型')
    parser.add_argument('--num_games', type=int, default=50, help='评估的游戏数量')
    parser.add_argument('--max_steps', type=int, default=50, help='每个episode的最大步数')
    parser.add_argument('--output', default='random_agent_results.json', help='结果输出文件')
    args = parser.parse_args()
    
    # 加载配置
    config = generic.load_config()
    
    # 运行评估
    results = evaluate_random_agent(
        config, 
        args.env_type, 
        num_games=args.num_games, 
        max_steps_per_episode=args.max_steps
    )
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {args.output}")

def init_xvfb(display_id=100):
    """Initialize virtual display for visualization"""
    _xvfb_proc = subprocess.Popen(
        ["Xvfb", f":{display_id}", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = f":{display_id}"
    return _xvfb_proc

if __name__ == "__main__":
    xvfb_proc = init_xvfb()
    main()
    xvfb_proc.terminate()

"""python simple_test.py"""