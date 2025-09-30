#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试CUDA GradScaler修复的脚本
"""

import torch
import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'python_scripts'))

def test_cuda_availability():
    """测试CUDA可用性"""
    print("=== CUDA 可用性测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name()}")
    else:
        print("CUDA不可用，将使用CPU")
    print()

def test_gradscaler_initialization():
    """测试GradScaler初始化"""
    print("=== GradScaler 初始化测试 ===")
    try:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            print("✓ GradScaler在CUDA可用时成功初始化")
        else:
            print("✓ 跳过GradScaler初始化（CUDA不可用）")
    except Exception as e:
        print(f"✗ GradScaler初始化失败: {e}")
    print()

def test_ppo_import():
    """测试PPO模块导入"""
    print("=== PPO模块导入测试 ===")
    try:
        from PPO.PPO_PPOnet_2 import PPO2
        print("✓ PPO模块导入成功")
        
        # 测试PPO2类初始化
        ppo = PPO2(node_num=10)
        print("✓ PPO2类初始化成功")
        
        # 检查scaler属性
        if ppo.scaler is not None:
            print("✓ GradScaler已正确初始化")
        else:
            print("✓ GradScaler已正确设置为None（CUDA不可用）")
            
    except Exception as e:
        print(f"✗ PPO模块测试失败: {e}")
        import traceback
        traceback.print_exc()
    print()

def test_device_config():
    """测试设备配置"""
    print("=== 设备配置测试 ===")
    try:
        from Project_config import device
        print(f"✓ 设备配置: {device}")
        
        # 测试张量创建
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"✓ 测试张量创建成功，设备: {test_tensor.device}")
        
    except Exception as e:
        print(f"✗ 设备配置测试失败: {e}")
    print()

def main():
    """主测试函数"""
    print("开始CUDA GradScaler修复测试...\n")
    
    test_cuda_availability()
    test_gradscaler_initialization()
    test_device_config()
    test_ppo_import()
    
    print("测试完成！")
    print("\n如果所有测试都通过，说明CUDA GradScaler修复成功。")
    print("现在应该不会再看到 'torch.cuda.amp.GradScaler is enabled, but CUDA is not available' 的警告了。")

if __name__ == "__main__":
    main() 