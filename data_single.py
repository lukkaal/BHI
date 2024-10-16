import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 读取数据文件
file_path = 'D:\BHI_Contest\BHI_Contest\Material Track-1\SyntheticData/User1.csv'  # 请替换为你实际的文件路径
df = pd.read_csv(file_path)

# 1. 处理缺失值（用0填充数值列的空值）
df.fillna(0, inplace=True)

# 2. 独热编码 'Activity_Type' 列
onehot_encoder = OneHotEncoder(sparse_output=False)
activity_encoded = onehot_encoder.fit_transform(df[['Activity_Type']])

# 3. 标准化数值特征
numerical_features = df[['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
                         'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors']]
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_features)

# 4. 合并处理后的数据
processed_data = np.hstack([numerical_scaled, activity_encoded])

# 5. 将数据转换为LSTM模型的输入格式 (num_samples, timesteps, features)
# 假设每个时间步是一个样本
X_processed = np.expand_dims(processed_data, axis=1)

# 6. 打印处理后的数据形状
print("处理后的数据形状：", X_processed.shape)
