import glob
import os
import re

import torch


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _next_log_file(dir_path: str, prefix: str) -> str:
    pattern = os.path.join(dir_path, f"{prefix}_*.json")
    existing = glob.glob(pattern)
    max_n = 0
    for file_path in existing:
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.json$", os.path.basename(file_path))
        if match:
            try:
                number = int(match.group(1))
                if number > max_n:
                    max_n = number
            except Exception:
                continue
    return os.path.join(dir_path, f"{prefix}_{max_n + 1}.json")


def _latest_single_ckpt(dir_path: str):
    """加载目录中最新的 single_hppo_*.ckpt。"""
    files = glob.glob(os.path.join(dir_path, "single_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(file_path: str) -> int:
        base_name = os.path.basename(file_path)
        match = re.search(r"single_hppo_(\d+)\.ckpt$", base_name)
        return int(match.group(1)) if match else -1

    selected = max(files, key=_num)
    return selected, _num(selected)


def _save_single_checkpoint(agent, ckpt_dir: str, total_episode: int):
    save_path = os.path.join(ckpt_dir, f"single_hppo_{total_episode}.ckpt")
    checkpoint = {
        "policy": agent.policy.state_dict(),
        "optimizer_hppo": agent.optimizer.state_dict(),
        "episode": total_episode,
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


# ===== 模型加载工具函数（提炼提高可读性） =====
def load_single_model(model_path: str, hppo_agent, ckpt_dir: str) -> int:
    """加载单智能体模型，优先指定路径；否则自动加载目录最新。"""
    episode_start = 0
    if model_path:
        try:
            ckpt = torch.load(model_path)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                hppo_agent.policy.load_state_dict(ckpt["policy"])
                if "optimizer_hppo" in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt["optimizer_hppo"])
                print(f"从指定模型加载: {model_path}，单智能体模型加载成功！")
                try:
                    episode_start = int(os.path.basename(model_path).split("_")[-1].split(".")[0])
                    print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
                except Exception:
                    pass
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print(f"从指定模型加载: {model_path}，单智能体模型加载成功！")
        except Exception as exc:
            print(f"指定模型加载失败: {exc}")
            episode_start = 0
        return episode_start

    # 未指定路径，查找目录最新
    selected_model, episode_start = _latest_single_ckpt(ckpt_dir)
    if selected_model:
        try:
            ckpt = torch.load(selected_model)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                hppo_agent.policy.load_state_dict(ckpt["policy"])
                if "optimizer_hppo" in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt["optimizer_hppo"])
                print("单智能体模型加载成功！")
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print("单智能体模型加载成功！")
        except Exception as exc:
            print(f"单智能体模型加载失败: {exc}")
            episode_start = 0
    else:
        print("未找到已保存的单智能体模型，从头开始训练")
        episode_start = 0
    return episode_start
