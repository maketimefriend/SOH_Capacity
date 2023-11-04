import csv
import os
from statistics import mean

def remove_and_split_columns(input_file, output_file_1, output_file_2, column_indices, split_column_index,split_column_index01):
    with open(input_file, 'r', errors='ignore') as file_in:
        reader = csv.reader(file_in)
        header = next(reader)  # 读取并保存文件头
        V_header = [i for i in range(1,97)]
        T_header = [i for i in range(1,49)]
        new_header = ['time','vehicle_state','charge_state','speed','distance','total_V','total_I','SOC','DC_DC','motor_number','Max_cell_V','Min_cell_V','Max_T','Min_T','upper_alarm','common_alarm','ave_V','ave_T']+V_header+T_header

        with open(output_file_1, 'w', newline='') as file_out_1, open(output_file_2, 'w', newline='') as file_out_2:
            writer_1 = csv.writer(file_out_1)
            writer_1.writerow(new_header)  # 写入文件1的文件头

            writer_2 = csv.writer(file_out_2)
            writer_2.writerow(new_header)  # 写入文件2的文件头

            for row in reader:
                new_row = [row[i] for i in range(len(header)) if i not in column_indices]  # 根据列索引删除指定列
                for split_index in split_column_index01:
                    split_data = row[split_index].split(":")[1:]  # 将数据根据逗号拆分，忽略冒号前面的数值
                    split_data = [item.split("_") for item in split_data]  # 将拆分后的数据再根据分号拆分为列表
                    flattened_data = [item for sublist in split_data for item in sublist]  # 将二维列表平铺成一维列表
                    new_row.extend(flattened_data)  # 将拆分后的数据添加到新行

                if row[split_column_index] == '1':
                    writer_1.writerow(new_row)  # 写入到输出文件1
                else:
                    writer_2.writerow(new_row)  # 写入到输出文件2


    print("CSV columns removed and splitted into two files.")

input_folder = 'C:/Users/18167/Desktop/data'
output_folder_1 = 'C:/Users/18167/Desktop/discharge'
output_folder_2 = 'C:/Users/18167/Desktop/charge'

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):  # 检查文件是否是CSV文件
        input_file = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]  # 获取文件名（不包含扩展名）

        # 构建输出文件的路径
        output_file_1 = os.path.join(output_folder_1, base_filename + '_discharge.csv')
        output_file_2 = os.path.join(output_folder_2, base_filename + '_charge.csv')

        a =[x for x in range(15, 43)]
        b = [y for y in range(55, 85)]
        column_indices01 = a + b + [0, 1, 5, 12, 13, 44, 45, 47, 48, 50, 51, 86, 87]  # 指定要删除的列索引列表
        column_indices02 =  set(column_indices01)
        column_indices = list(column_indices02)
        split_column_index = 14  # 指定数据分割的列索引
        split_column_index01 = [85, 88]

        # 调用 remove_and_split_columns 函数处理当前文件
        remove_and_split_columns(input_file, output_file_1, output_file_2, column_indices, split_column_index, split_column_index01)