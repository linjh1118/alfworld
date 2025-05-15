from multiprocessing import Pool
import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import subprocess
import random
import sys
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_tw_env import TASK_TYPES
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
if len(sys.argv) == 1:
    sys.argv.append('/workspace/linjh/rl/Game-Easy-R1/alfworld/configs/base_config.yaml')

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

def process_single_game(args):
    """
    处理单个游戏的评估函数
    """
    game_file, config, train_eval, display_id = args
    
    # 初始化虚拟显示，每个进程需要自己的显示
    xvfb_proc = init_xvfb(display_id)
    
    try:
        # 创建环境
        env = FixedGameAlfredThorEnv(config, train_eval=train_eval, fixed_game_file=game_file)
        env = env.init_env(batch_size=1)
        
        try:
            # 重置环境
            obs, info = env.reset()
            
            # 加载任务数据
            game_path = info["extra.gamefile"][0]
            traj_json_path = os.path.join(game_path, "traj_data.json")
            
            with open(traj_json_path, 'r') as f:
                traj_data = json.load(f)
            
            # 记录任务类型
            task_type = traj_data['task_type']
            
            # 模拟任务执行
            success = np.random.choice([True, False], p=[0.5, 0.5])
            
            return {
                'task_type': task_type,
                'success': success
            }
            
        finally:
            env.close()
            
    finally:
        try:
            xvfb_proc.terminate()
        except:
            pass

def evaluate_env(config, train_eval, display_id=100, n_sample_game=None):
    """
    并行评估特定分布的环境性能
    """
    # 首先创建一个临时环境来获取所有游戏文件列表
    temp_env = AlfredThorEnv(config, train_eval=train_eval)
    game_files = temp_env.json_file_list
    temp_env.close()
    if n_sample_game:
        game_files = random.sample(game_files, min(n_sample_game, len(game_files)))
    
    print(f"\nEvaluating {train_eval} set with {len(game_files)} games")
    
    # 准备并行处理的参数
    num_processes = 8
    # 为每个进程分配不同的显示ID
    process_args = [(game_file, config, train_eval, display_id + i) 
                   for i, game_file in enumerate(game_files)]

    # 使用进程池进行并行处理
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_single_game, process_args),
            total=len(game_files),
            desc=f"Processing {train_eval} games"
        ))
        
    # 统计结果
    task_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    total_episodes = 0
    total_success = 0
    
    # 统计结果
    for result in results:
        task_type = result['task_type']
        success = result['success']
        
        task_stats[task_type]['total'] += 1
        total_episodes += 1
        
        if success:
            task_stats[task_type]['success'] += 1
            total_success += 1
    
    # 计算总体成功率
    overall_success_rate = total_success / total_episodes if total_episodes > 0 else 0
    
    # 计算每种任务类型的成功率
    task_success_rates = {}
    for task_type, stats in task_stats.items():
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        task_success_rates[task_type] = {
            'success_rate': success_rate,
            'total_episodes': stats['total']
        }
    
    return overall_success_rate, task_success_rates

def print_results(eval_type, overall_rate, task_rates):
    """
    打印评估结果
    """
    print(f"\n=== {eval_type} Results ===")
    print(f"Overall Success Rate: {overall_rate:.2%}")
    print("\nTask-specific Success Rates:")
    for task_type, stats in task_rates.items():
        print(f"{task_type}:")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Total Episodes: {stats['total_episodes']}")

def main():
    # 加载配置
    config = generic.load_config()
    
    # 设置基础display ID，为两种评估预留足够的display ID空间
    base_display_id_in_dist = 100
    base_display_id_out_dist = 200
    
    print("\nStarting evaluation...")
    
    # 评估分布内测试集
    print("\n=== Evaluating In-Distribution Set ===")
    overall_in_dist, task_rates_in_dist = evaluate_env(
        config,
        'eval_in_distribution',
        display_id=base_display_id_in_dist,
        n_sample_game=16
    )
    
    # 清理一下进程和资源
    import gc
    gc.collect()
    
    # 评估分布外测试集
    print("\n=== Evaluating Out-of-Distribution Set ===")
    overall_out_dist, task_rates_out_dist = evaluate_env(
        config,
        'eval_out_of_distribution',
        display_id=base_display_id_out_dist,
        n_sample_game=16
    )
    
    # 打印结果
    print("\n=== Final Results ===")
    print_results("In-Distribution", overall_in_dist, task_rates_in_dist)
    print_results("Out-of-Distribution", overall_out_dist, task_rates_out_dist)
    
    # 保存结果到CSV，添加时间戳
    from datetime import datetime
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = model_id
    results = []
    
    for eval_type, overall_rate, task_rates in [
        ("In-Distribution", overall_in_dist, task_rates_in_dist),
        ("Out-of-Distribution", overall_out_dist, task_rates_out_dist)
    ]:
        row = {
            'Evaluation Type': eval_type,
            'Overall Success Rate': f"{overall_rate:.2%}",
            'Model ID': model_id
        }
        for task_type, stats in task_rates.items():
            row[f"{task_type} Success Rate"] = f"{stats['success_rate']:.2%}"
            row[f"{task_type} Episodes"] = stats['total_episodes']
        results.append(row)
    
    # 保存到CSV
    output_path = f'evaluation_results_{timestamp}.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults have been saved to {output_path}")
    
    try:
        os.system(f'rich {output_path}')
    except Exception as e:
        print(f"Could not display results with rich: {e}")
        print("Results are still saved in the CSV file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        # 确保清理所有子进程
        import signal
        os.kill(0, signal.SIGTERM)
    except Exception as e:
        print(f"\nError occurred: {e}")
        # 确保清理所有子进程
        import signal
        os.kill(0, signal.SIGTERM)
    finally:
        # 最后的清理工作
        print("\nCleaning up resources...")

if __name__ == "__main__":
    main()