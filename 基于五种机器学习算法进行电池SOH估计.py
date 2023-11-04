import os
import shutil
import joblib
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

matplotlib.rcParams['font.sans-serif']=['SimHei'] # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False
# 源文件夹和目标文件夹路径
source_folder = 'C:/Users/18167/Desktop/SOH'  # 源文件夹路径
target_folder = 'C:/Users/18167/Desktop/SOH不同算法预测结果'  # 目标文件夹路径
# target_folder1 = 'C:/Users/18167/Desktop/SOH_XGboost算法预测结果'  # 目标文件夹路径
# target_folder2 = 'C:/Users/18167/Desktop/SOH_线性回归算法预测结果'  # 目标文件夹路径
# target_folder3 = 'C:/Users/18167/Desktop/SOH_支持向量机算法预测结果'  # 目标文件夹路径
# target_folder4 = 'C:/Users/18167/Desktop/深度学习算法预测结果'  # 目标文件夹路径
# 遍历源文件夹中的文件
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.csv'):  # 只处理以.csv结尾的文件
            # 构造源文件和目标文件的路径
            source_file = os.path.join(root, file)
            base_filename = os.path.splitext(file)[0]  # 获取文件名（不包含扩展名）
            target_file = os.path.join(target_folder,base_filename+'.PNG')  # 构建输出文件的路径1
            # target_file1 = os.path.join(target_folder1,base_filename+'.PNG')  # 构建输出文件的路径2
            # target_file2 = os.path.join(target_folder2, base_filename + '.PNG')  # 构建输出文件的路径3
            # target_file3 = os.path.join(target_folder3, base_filename + '.PNG')  # 构建输出文件的路径4
            # target_file4 = os.path.join(target_folder4, base_filename + '.PNG')  # 构建输出文件的路径4
            # 将最后一列作为目标变量，其余列作为特征变量

            # 导入数据
            df = pd.read_csv(source_file)
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]

            # 定义k-fold交叉验证的折数
            kfold = 5
            RMSE = []
            MAE = []
            R2 = []
            Runtime = []

            # 随机森林算法
            start_time = time.time()
            rf_model = RandomForestRegressor()
            kf = KFold(n_splits=kfold)
            rmse_list = []
            mae_list = []
            r2_list = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)
            end_time = time.time()
            runtime = end_time - start_time
            avg_rmse1 = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_r2 = np.mean(r2_list)

            RMSE.append(avg_rmse1)
            MAE.append(avg_mae)
            R2.append(avg_r2)
            Runtime.append(runtime)

            print("Random Forest:")
            print("Average RMSE:", avg_rmse1)
            print("Average MAE:", avg_mae)
            print("Average R^2:", avg_r2)
            print("Runtime:", runtime)
            print()

            # XGBoost算法
            start_time = time.time()
            xgb_model = xgb.XGBRegressor()
            rmse_list = []
            mae_list = []
            r2_list = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)

            end_time = time.time()
            runtime = end_time - start_time
            avg_rmse2 = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_r2 = np.mean(r2_list)

            RMSE.append(avg_rmse2)
            MAE.append(avg_mae)
            R2.append(avg_r2)
            Runtime.append(runtime)
            print("XGBoost:")
            print("Average RMSE:", avg_rmse2)
            print("Average MAE:", avg_mae)
            print("Average R^2:", avg_r2)
            print("Runtime:", runtime)
            print()

            # 支持向量机SVR算法
            start_time = time.time()
            svr_model = SVR()
            rmse_list = []
            mae_list = []
            r2_list = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                svr_model.fit(X_train, y_train)
                y_pred = svr_model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)

            end_time = time.time()
            runtime = end_time - start_time
            avg_rmse3 = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_r2 = np.mean(r2_list)

            RMSE.append(avg_rmse3)
            MAE.append(avg_mae)
            R2.append(avg_r2)
            Runtime.append(runtime)
            print("Support Vector Machine (SVR):")
            print("Average RMSE:", avg_rmse3)
            print("Average MAE:", avg_mae)
            print("Average R^2:", avg_r2)
            print("Runtime:", runtime)
            print()

            # 线性回归算法
            start_time = time.time()
            lr_model = LinearRegression()
            rmse_list = []
            mae_list = []
            r2_list = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                lr_model.fit(X_train, y_train)
                y_pred = lr_model.predict(X_test)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)

            end_time = time.time()
            runtime = end_time - start_time
            avg_rmse4 = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_r2 = np.mean(r2_list)

            RMSE.append(avg_rmse4)
            MAE.append(avg_mae)
            R2.append(avg_r2)
            Runtime.append(runtime)
            print("Linear Regression:")
            print("Average RMSE:", avg_rmse4)
            print("Average MAE:", avg_mae)
            print("Average R^2:", avg_r2)
            print("Runtime:", runtime)

            # 神经网络算法
            start_time = time.time()
            features = df.iloc[:, 1:].values
            labels = df.iloc[:, 0].values
            scaler = MinMaxScaler()
            features_normalized = scaler.fit_transform(features)
            rmse_list = []
            mae_list = []
            r2_list = []
            for train_index, test_index in kf.split(features_normalized):
                train_features, test_features = features_normalized[train_index], features_normalized[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                nn_model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(train_features.shape[1],)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                nn_model.compile(optimizer='adam', loss='mean_squared_error')
                history = nn_model.fit(train_features, train_labels, validation_data=(test_features, test_labels),
                                    epochs=350)
                y_pred = nn_model.predict(test_features).flatten()
                rmse = mean_squared_error(test_labels, y_pred, squared=False)
                mae = mean_absolute_error(test_labels, y_pred)
                r2 = r2_score(test_labels, y_pred)
                rmse_list.append(rmse)
                mae_list.append(mae)
                r2_list.append(r2)

            end_time = time.time()
            runtime = end_time - start_time
            avg_rmse5 = np.mean(rmse_list)
            avg_mae = np.mean(mae_list)
            avg_r2 = np.mean(r2_list)

            RMSE.append(avg_rmse5)
            MAE.append(avg_mae)
            R2.append(avg_r2)
            Runtime.append(runtime)
            print("Neural Network:")
            print("Average RMSE:", avg_rmse5)
            print("Average MAE:", avg_mae)
            print("Average R^2:", avg_r2)
            print("Runtime:", runtime)


            y_pred_rf = rf_model.predict(X)
            y_pred_xgb = xgb_model.predict(X)
            y_pred_svr = svr_model.predict(X)
            y_pred_lr = lr_model.predict(X)
            y_pred_nn = nn_model.predict(features_normalized).flatten()

            #不同算法模型预测值
            # 行名称和列名称
            rows = df['循环次数']
            cols = ['循环次数','日历时间','行驶里程','电池容量真实值',"随机森林算法预测值", "XG_Boost算法预测值", "支持向量机算法预测值", "线性回归算法预测值", "神经网络算法预测值"]
            # 桌面路径
            ACC_path = os.path.expanduser(r'C:\Users\18167\Desktop\model_data')
            # 创建完整的文件路径
            file_path = os.path.join(ACC_path, "{}不同算法电池容量预测值.csv".format(base_filename))
            # 创建并写入CSV文件
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入列名称
                writer.writerow(cols)
                # 写入数据
                for i in range(len(rows)):
                    writer.writerow([rows[i],df['日历时间'][i],df['行驶里程'][i],df['电池容量'][i], y_pred_rf[i],y_pred_xgb[i], y_pred_svr[i], y_pred_lr[i], y_pred_nn[i]])

            # 不同算法模型预测精度
            rows = ["Random Forest", "XG_Boost", "SVR", "Linear Regression", "Neural Network"]
            cols = ["模型名称", "RMSE", "MAE", "R^2", "Runtime"]
            # 桌面路径
            ACC_path = os.path.expanduser(r'C:\Users\18167\Desktop\model_acc')
            # 创建完整的文件路径
            file_path = os.path.join(ACC_path, "{}不同算法评估分值.csv".format(base_filename))
            # 创建并写入CSV文件
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入列名称
                writer.writerow(cols)
                # 写入数据
                for i in range(len(rows)):
                    writer.writerow([rows[i], RMSE[i],MAE[i], R2[i], Runtime[i]])

            # 绘制预测结果图像
            data_list = [y, y_pred_rf, y_pred_xgb, y_pred_svr, y_pred_lr, y_pred_nn]  # 假设有六个列表数据
            color_list = ['green', 'red', 'blue', 'orange', 'purple', 'yellow']  # 每个列表数据对应的颜色
            marker_list = ['o', 's', 'D', 'v', '^', 'p']  # 每个列表数据对应的形状
            label_list = ['电池容量真实值', '随机森林算法预测值', 'XG_boost算法预测值', '线性回归算法预测值', '支持向量机SVR算法预测值',
                          '神经网络算法预测值']  # 每个列表数据的标签
            fig = plt.figure(figsize=(25, 15))
            plt.tick_params(labelsize=30)
            counter = 0
            for data, color, marker, label in zip(data_list, color_list, marker_list, label_list):
                for i, (a, b) in enumerate(zip(df['日历时间'] / 3600 / 24, data)):
                    if i % 20 == 0:
                        if i == 0:
                            plt.scatter(a, b, s=100, label=label, color=color, marker=marker)
                        else:
                            plt.scatter(a, b, s=100, color=color, marker=marker)
                        if color == 'green' and counter%2==0:
                            plt.text(int(a), b + 0.3, f'{b:.1f}', ha='center', va='bottom', fontsize=20)
                    counter += 1

            plt.legend(fontsize=25, loc='upper right')
            plt.xlabel('日历时间/day', fontsize=35)
            plt.ylabel('电池容量/A.h', fontsize=35)
            plt.ylim(100, 138)
            plt.title('{}电池日历时间'.format(base_filename), fontsize=40)
            plt.savefig(target_file, dpi=600)
            plt.close()  # 关闭当前图形

            # 指定模型文件保存路径为桌面
            save_path = os.path.expanduser(r"C:/Users/18167/Desktop/model")

            min_rmse = min(avg_rmse1, avg_rmse2, avg_rmse3, avg_rmse4, avg_rmse5)
            if min_rmse == avg_rmse1:
                rf_model.fit(X, y)
                print("Random Forest is selected.")
                model_path = os.path.join(save_path, "{}random_forest_model.pkl".format(base_filename))
                joblib.dump(rf_model, model_path)
            elif min_rmse == avg_rmse2:
                xgb_model.fit(X, y)
                print("XGBoost is selected.")
                model_path = os.path.join(save_path, "{}xgboost_model.pkl".format(base_filename))
                joblib.dump(xgb_model, model_path)
            elif min_rmse == avg_rmse3:
                svr_model.fit(X, y)
                print("Support Vector Machine (SVR) is selected.")
                model_path = os.path.join(save_path, "{}svr_model.pkl".format(base_filename))
                joblib.dump(svr_model, model_path)
            elif min_rmse == avg_rmse4:
                lr_model.fit(X, y)
                print("Linear Regression is selected.")
                model_path = os.path.join(save_path, "{}linear_regression_model.pkl".format(base_filename))
                joblib.dump(lr_model, model_path)
            else:
                nn_model.fit(features_normalized, labels)
                print("Neural Network is selected.")
                model_path = os.path.join(save_path, "neural_network_model.h5".format(base_filename))
                nn_model.save(model_path)

            print("The trained model has been saved at: ", model_path)

