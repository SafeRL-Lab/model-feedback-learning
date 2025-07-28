import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

# 神经网络定义
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(11, 64)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(64, 6)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

model = NeuralNet()
model_path = "/home/gushangding/vt_data_25/data_efficiency/semi_example/lam_target_icml/constraint_input/domain_randomization/model/emulator/p3_r_input_train_emulator_icml_domain_r.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

search_ranges = {
    0: (5, 38), 1: (4000, 25000), 2: (0, 8000), 3: (0, 600),
    4: (0, 80), 5: (0, 66), 6: (0, 20), 7: (0, 50),
    8: (10, 70), 9: (950, 1050), 10: (10, 55),
}
pbounds = {'x%d' % i: v for i, v in search_ranges.items()}

target = np.array([2260, 110, 360, 200, 10, 200], dtype=np.float32)

# 用于保存所有采样结果
history = []

def objective(**params):
    x = np.array([params['x%d' % i] for i in range(11)], dtype=np.float32).reshape(1, -1)
    x_tensor = torch.tensor(x)
    with torch.no_grad():
        y = model(x_tensor).numpy().flatten()
    etch_depth, etch_rate, mask_remaining, top_cd, delta_cd, bow_cd = y
    constraint = (
        (2250 <= etch_depth <= 2750) and
        (etch_rate >= 100) and
        (mask_remaining >= 350) and
        (190 <= top_cd <= 210) and
        (-15 <= delta_cd <= 15) and
        (190 <= bow_cd <= 210)
    )
    loss = np.mean((y - target) ** 2)
    # 收集采样结果
    result = {
        **{f'x{i}': float(x.flatten()[i]) for i in range(11)},
        **{f'y{i}': float(y[i]) for i in range(6)},
        'loss': loss,
        'constraint': constraint
    }
    history.append(result)
    # print 
    print(f"params={np.round(x.flatten(), 2)}, y={np.round(y, 2)}, loss={loss:.2f}, constraint={constraint}")
    if constraint:
        return etch_depth
    else:
        return -1e6

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=45,
    verbose=2
)
optimizer.maximize(init_points=10, n_iter=1000)

print("Best params:", optimizer.max)

# 保存结果到 CSV
df = pd.DataFrame(history)
save_path = "/home/gushangding/vt_data_25/data_efficiency/semi_example/lam_target_icml/Bayesian_optimization/results/bo_eval_history.csv"
df.to_csv(save_path, index=False)
print(f"All evaluation results saved to {save_path}")
