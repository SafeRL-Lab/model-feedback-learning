from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np

# 1. 目标输出的“目标值”
target = np.array([2260, 110, 360, 200, 10, 200], dtype=np.float32)

# 输入归一化参数
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
input_min = np.array([v[0] for v in pbounds.values()], dtype=np.float32)
input_max = np.array([v[1] for v in pbounds.values()], dtype=np.float32)
input_range = input_max - input_min

# 输出归一化参数
output_min = np.array([2250, 99, 349, 190, -15, 190], dtype=np.float32)
output_max = np.array([2750, 300, 450, 210, 15, 210], dtype=np.float32)
output_range = output_max - output_min
target_normalized = (target - output_min) / output_range

# 采样结果记录
history = []

# 黑盒目标函数，输入归一化/反归一化
def black_box_function_norm(x_norm):
    # x_norm: 归一化输入 [0, 1]
    x = x_norm * input_range + input_min
    # 构造6个输出
    y = np.zeros(6)
    y[0] = 2200 + (x[1] / 100) + np.random.normal(0, 10)         # Etch depth
    y[1] = 100 + (x[0] / 10) + np.random.normal(0, 2)            # Etch rate
    y[2] = 360 + (x[2] / 1000) + np.random.normal(0, 8)          # Mask remaining
    y[3] = 200 + (x[3] / 2000) + np.random.normal(0, 1)          # Top CD
    y[4] = 10 + (x[4] / 40) + np.random.normal(0, 1)             # ΔCD
    y[5] = 200 + (x[5] / 20) + np.random.normal(0, 1)            # Bow CD
    return y

# 包装函数：接收原始参数（bayes_opt需要），内部转归一化
def logging_black_box_function(**params):
    x = np.array([params[f'x{i}'] for i in range(11)], dtype=np.float32)
    x_norm = (x - input_min) / input_range
    y = black_box_function_norm(x_norm)
    y_norm = (y - output_min) / output_range
    loss = np.mean((y_norm - target_normalized) ** 2)
    etch_depth, etch_rate, mask_remaining, top_cd, delta_cd, bow_cd = y

    constraint = (
        (2250 <= etch_depth <= 2750) and
        (etch_rate >= 100) and
        (mask_remaining >= 350) and
        (190 <= top_cd <= 210) and
        (-15 <= delta_cd <= 15) and
        (190 <= bow_cd <= 210)
    )
    result = {
        **{f'x{i}': float(x[i]) for i in range(11)},
        **{f'x_norm{i}': float(x_norm[i]) for i in range(11)},
        **{f'y{i}': float(y[i]) for i in range(6)},
        **{f'y_norm{i}': float(y_norm[i]) for i in range(6)},
        'loss': loss,
        'constraint': constraint
    }
    history.append(result)
    print(f"params={np.round(x, 2)}, y={np.round(y,2)}, loss={loss:.4f}, constraint={constraint}")
    return etch_depth if constraint else -1e6

# BO主流程
optimizer = BayesianOptimization(
    f=logging_black_box_function,
    pbounds=pbounds,
    random_state=45,
    verbose=2
)
optimizer.maximize(init_points=100, n_iter=100)

# 输出最优
print("\nBest parameters found:")
print(optimizer.max)

# 保存所有采样历史到 CSV
df = pd.DataFrame(history)
save_path = "/home/gushangding/vt_data_25/data_efficiency/semi_example/lam_target_icml/Bayesian_optimization/results/bo_gp_history_with_loss_constraint_norm.csv"
df.to_csv(save_path, index=False)
print(f"All evaluation history saved to {save_path}")
