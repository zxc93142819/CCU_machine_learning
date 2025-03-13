import numpy as np
import argparse
from DataLoader import DataLoader
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import time

def pocket(DataLoader):
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  
    """
    max_iters = 10000
    weight_matrix = np.zeros(3)
    s = time.time()

    data = np.array(DataLoader.data)
    label = np.array(DataLoader.label)
    
    best_weight_matrix = weight_matrix.copy()
    best_error = len(label)  # 最差情況：全部錯誤
    iter = 0
    
    while iter < max_iters:
        dot = np.dot(data, weight_matrix)
        
        # 找出錯誤分類的索引
        wrong_indices = np.where((dot >= 0) & (label == -1) | (dot <= 0) & (label == 1))[0]
        
        if len(wrong_indices) == 0:  # 已經完全分割
            break
        
        iter += 1
        
        # 隨機選一個錯誤的樣本來更新權重
        wrong_index = np.random.choice(wrong_indices)
        new_weight_matrix = weight_matrix + label[wrong_index] * data[wrong_index]

        # 計算新權重的錯誤率
        new_dot = np.dot(data, new_weight_matrix)
        new_wrong = np.sum((new_dot >= 0) & (label == -1) | (new_dot <= 0) & (label == 1))

        # 只在錯誤數減少時更新最佳權重
        if new_wrong < best_error:
            best_error = new_wrong
            best_weight_matrix = new_weight_matrix.copy()
        
        # **強制更新機制**（避免卡住在局部最優）
        if iter % 1000 == 0:  
            weight_matrix = new_weight_matrix.copy()

    # 計算最終的分類準確率
    final_dot = np.dot(data, best_weight_matrix)
    final_wrong = np.sum((final_dot >= 0) & (label == -1) | (final_dot <= 0) & (label == 1))
    accuracy = (1 - (final_wrong / len(data))) * 100

    print(f"Accuracy: {accuracy:.2f}%")
    e = time.time()
    print("Execution time = %f" % (e - s))
    
    return best_weight_matrix

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = pocket(DataLoader=Loader)

    # This part is for plotting the graph
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(Loader.cor_x_pos, Loader.cor_y_pos,
                c='b', label='pos data')
    plt.scatter(Loader.cor_x_neg, Loader.cor_y_neg,
                c='r', label='neg data')

    x = np.linspace(-1000, 1000, 100)
    # This is the base line
    y1 = 3*x+5
    # This is your split line
    y2 = (updated_weight[1]*x + updated_weight[0]) / (-updated_weight[2])
    plt.plot(x, y1, 'g', label='base line', linewidth='1')
    plt.plot(x, y2, 'y', label='split line', linewidth='1')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)