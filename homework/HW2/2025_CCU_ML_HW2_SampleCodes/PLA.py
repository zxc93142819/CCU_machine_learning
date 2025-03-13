import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import time


def PLA(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the PLA here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    s = time.time()
    ############ START ##########
    sum_iter = 0
    times = 6
    for t in range(1 , times + 1) :
        # count iteration
        iter = 0
        data = np.array(DataLoader.data)
        label = np.array(DataLoader.label)
        # 先找一個data當initial weight
        random_index = np.random.randint(len(data))
        weight_matrix = data[random_index]
        while(True) :
            dot = np.dot(data , weight_matrix)
            # 找出錯誤分類的索引
            wrong_indices = np.where((dot >= 0) & (label == -1) | (dot <= 0) & (label == 1))[0]
            
            if len(wrong_indices) == 0:  # 已經完全分割
                break
            
            iter += 1
            
            # 隨機選一個錯誤的樣本來更新權重
            wrong_index = wrong_indices[0]
            weight_matrix = weight_matrix + label[wrong_index] * data[wrong_index]

        print("第" , t , "次： iterations次數為" , iter , "次")
        sum_iter += iter
    print("平均iteration次數為" , sum_iter / times , "次")
    ############ END ############
    e = time.time()
    print("ex time : %f" % (e-s))
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = PLA(DataLoader=Loader)

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
