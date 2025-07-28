from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np

# 1. 目标输出的“目标值”
target = np.array([2260, 110, 360, 200, 10, 200], dtype=np.float32)

# 2. 定义输入参数空间
pbounds = {
    'x0': (5, 38),
    'x1': (4000, 25000),
    'x2': (0, 8000),
    'x3': (0, 600),
    'x4': (0, 80),
    'x5': (0, 66),
    'x6': (0, 20),
    'x7': (0, 50),
    'x8': (10, 70),
    'x9': (950, 1050),
    'x10': (10, 55),
}

# 3. 用于保存每次评估的结果
history = []

# 4. 目标函数
def black_box_function(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    """
    假定输出y0~y5是输入线性加高斯噪声生成的例子（可替换成真实实验/仿真）
    """
    x = np.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    # 构造6个输出
    y = np.zeros(6)
    # 示例 y0~y5
    y[0] = 2200 + (x1 / 100) + np.random.normal(0, 10)         # Etch depth
    y[1] = 100 + (x0 / 10) + np.random.normal(0, 2)            # Etch rate
    y[2] = 360 + (x2 / 1000) + np.random.normal(0, 8)          # Mask remaining
    y[3] = 200 + (x3 / 2000) + np.random.normal(0, 1)          # Top CD
    y[4] = 10 + (x4 / 40) + np.random.normal(0, 1)             # ΔCD
    y[5] = 200 + (x5 / 20) + np.random.normal(0, 1)            # Bow CD
    return y

# 5. 日志包装目标函数
def logging_black_box_function(**params):
    y = black_box_function(**params)
    loss = np.mean((y - target) ** 2)
    etch_depth, etch_rate, mask_remaining, top_cd, delta_cd, bow_cd = y
    # 约束条件
    constraint = (
        (2250 <= etch_depth <= 2750) and
        (etch_rate >= 100) and
        (mask_remaining >= 350) and
        (190 <= top_cd <= 210) and
        (-15 <= delta_cd <= 15) and
        (190 <= bow_cd <= 210)
    )
    # 收集采样结果
    result = {
        **{f'x{i}': float(params[f'x{i}']) for i in range(11)},
        **{f'y{i}': float(y[i]) for i in range(6)},
        'loss': loss,
        'constraint': constraint
    }
    history.append(result)
    print(f"params={[round(params[f'x{i}'],2) for i in range(11)]}, y={np.round(y,2)}, loss={loss:.2f}, constraint={constraint}")
    # 返回 Etch depth，如果不满足约束则返回极低值
    return etch_depth if constraint else -1e6

# 6. 贝叶斯优化主流程
optimizer = BayesianOptimization(
    f=logging_black_box_function,
    pbounds=pbounds,
    random_state=45,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=50)

# 7. 输出最优结果
print("\nBest parameters found:")
print(optimizer.max)

# 8. 保存采样历史到 CSV
df = pd.DataFrame(history)
save_path = "/home/gushangding/vt_data_25/data_efficiency/semi_example/lam_target_icml/Bayesian_optimization/results/bo_gp_history_with_loss_constraint.csv"
df.to_csv(save_path, index=False)
print(f"All evaluation history saved to {save_path}")
