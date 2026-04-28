class TrainingManager:
    """
    单智能体学习节奏管理。
    核心思想：抓取和抬腿都先积累经验，再按统一间隔学习。
    """

    def __init__(self):
        self.shared_episodes = 0

        # 统一学习间隔，避免同一智能体在不同阶段过于频繁更新
        self.shared_learn_interval = 3

        print("【训练管理器初始化】")
        print(f"  单智能体学习间隔: {self.shared_learn_interval}个episodes")

    def should_learn_shared(self) -> bool:
        """决定是否执行单智能体学习"""
        result = (self.shared_episodes % self.shared_learn_interval == 0) and (self.shared_episodes > 0)
        return result

    def increment_shared(self):
        """单智能体episode计数加1"""
        self.shared_episodes += 1

    def get_status(self) -> str:
        """获取当前训练状态"""
        return f"[TrainingManager] Shared:{self.shared_episodes}"
