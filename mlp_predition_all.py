import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# 设置不使用科学计数法
np.set_printoptions(precision=2, suppress=True)

# 加载保存的模型和标准化器
corr_mlp = joblib.load('corrosion/model/corr/mlp_model.pkl')
corr_scaler_X = joblib.load('corrosion/model/corr/scaler_X.pkl')

mech_mlp = joblib.load('corrosion/model/mech/mlp_model.pkl')
mech_scaler_X = joblib.load('corrosion/model/mech/scaler_X.pkl')

data = np.load('corrosion/力学性能替换NaN后.npy')
y = data[:, -2:]
y_mean = np.mean(y, axis=0, keepdims=True)
y_std = np.std(y, axis=0, keepdims=True)

feature_names = [
    "Mn", "C", "Al", "Cr", "V",
    "Nb", "Ti", "Cu", "Mo", "Ni",
    "Sn", "Sb", "Si", "P", "晶粒尺寸",
    "实验条件", "腐蚀时间"
]

# 设置特征范围和步长
feature_range = [(0.5, 1.5), (0, 0.1), (0, 0), (0, 5), (0, 0), (0, 0.1), (0, 0), 
                  (0, 0.5), (0, 0), (0, 3), (0, 0), (0, 0), (0.1, 0.5), (0.005, 0.005), (15, 15)]
feature_steps = [0.1, 0.02, 0, 0.5, 0, 0.02, 0, 0.1, 0, 0.2, 0, 0.15, 0.1, 0, 1]

# 固定特征值
fixed_feature_17 = 72

# 进行一万次推理
num_predictions = 100000

# 打开文件用于写入
with open('corrosion/predition/all/predictions_all.txt', 'w') as f:
    for i in range(num_predictions):
        # 生成新的特征集，包括随机选择的第16个特征和固定特征
        corr_new_features = np.zeros((1, 17))
        for j in range(15):  # 为前15个特征生成值
            # 生成从下界开始，步长为 feature_steps[j]，直到不超过上界的值的数组
            if feature_range[j][0] == feature_range[j][1]:
                corr_new_features[0, j] = feature_range[j][0]
                continue
            possible_values = np.arange(feature_range[j][0], feature_range[j][1], feature_steps[j])
            # 从这个数组中随机选择一个值
            corr_new_features[0, j] = np.random.choice(possible_values)

        corr_new_features[0, 15] = np.random.choice([2, 2, 2, 2])
        corr_new_features[0, 16] = fixed_feature_17

        additional_value = 100 - np.sum(corr_new_features[0, :14])
        additional_value_formatted = f"{additional_value:.2f}"
        
        # 标准化新的特征集
        corr_new_features_scaled = corr_scaler_X.transform(corr_new_features)
        # print(corr_new_features.shape)
        
        # 使用模型进行预测
        corr_prediction = corr_mlp.predict(corr_new_features_scaled)

        mech_new_features = corr_new_features[0, :15]
        # print(mech_new_features.shape)
        mech_new_features_scaled = mech_scaler_X.transform(mech_new_features.reshape(1, -1))

        mech_prediction = mech_mlp.predict(mech_new_features_scaled)

        mech_inverse_prediction = mech_prediction * y_std + y_mean
        
        # 按特征名打印特征集和对应的预测值
        feature_output = ", ".join([f"{name}: {corr_new_features[0, i]:.2f}" for i, name in enumerate(feature_names)])
        output_line = f'Prediction: {i+1}, Fe: {additional_value_formatted}, {feature_output}, 失重率: {corr_prediction[0]:.4f}, 抗拉强度: {mech_inverse_prediction[0][0]:.1f}, 延伸率: {mech_inverse_prediction[0][1]:.1f}\n'
        
        # 打印到控制台
        print(output_line.strip())
        
        # 写入到文件中
        f.write(output_line)

# 确保文件路径正确，如果需要在特定目录下保存文件，需要指定完整路径