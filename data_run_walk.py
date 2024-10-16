import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import glob
import os

# 文件夹路径，假设所有CSV文件都在此文件夹中
folder_path = 'D:\BHI_Contest\BHI_Contest\Material Track-1\SyntheticData'

# 获取文件夹中所有CSV文件的路径
file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

# 初始化一个空的DataFrame，用于存放所有文件的数据
all_data = pd.DataFrame()

# 逐个读取文件并合并数据
for file_path in file_paths:
    # 读取CSV文件
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)

    # 过滤 'Activity_Type' 列中只包含 'running' 和 'walking' 的行
    df_filtered = df[df['Activity_Type'].isin(['Running', 'Walking'])]

    # 将过滤后的数据添加到all_data中
    all_data = pd.concat([all_data, df_filtered], ignore_index=True)

# 查看合并后的数据
print(f"合并后的数据形状: {all_data[:10]}")

# 2. 独热编码 'Activity_Type' 列
onehot_encoder = OneHotEncoder(sparse_output=False)
activity_encoded = onehot_encoder.fit_transform(all_data[['Activity_Type']])

# 3. 标准化数值特征
numerical_features = all_data[['Heart rate___beats/minute', 'Calories burned_kcal', 'Exercise duration_s',
                         'Sleep duration_minutes', 'Sleep type duration_minutes', 'Floors climbed___floors']]
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_features)

# 4. 合并处理后的数据
processed_data = np.hstack([numerical_scaled, activity_encoded])

# 5. 将数据转换为LSTM模型的输入格式 (num_samples, timesteps, features)
# 假设每个时间步是一个样本
X_processed = np.expand_dims(processed_data, axis=1)

# 6. 打印处理后的数据形状
print("处理后的数据形状：", X_processed[:5])
