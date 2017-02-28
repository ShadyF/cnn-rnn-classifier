import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    labels_list = []
    top_1_list = []
    top_5_list = []
    with open('./cnn_rnn_results.txt', 'r') as f:
        # Get rid of first line
        f.readline()

        for line in f:
            line_list = line.strip().split()
            label, top_1, top_5 = line_list
            labels_list.append(label)
            top_1_list.append(float(top_1))
            top_5_list.append(float(top_5))

    index = [i for i in range(len(labels_list))]

    print("Top 1 Average: {}%".format(np.average(top_1_list)))
    print("Top 5 Average: {}%".format(np.average(top_5_list)))

    plt.figure(1, figsize=(20, 10))
    plt.bar(index, top_1_list)
    plt.suptitle('Top-1 Accuracy per label')

    plt.ylabel('Probability (%)')
    plt.xlabel('Label')

    plt.xticks(index, labels_list, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)

    plt.figure(2, figsize=(20, 10))
    plt.bar(index, top_5_list)
    plt.suptitle('Top-5 Accuracy per label')

    plt.ylabel('Probability (%)')
    plt.xlabel('Label')

    plt.xticks(index, labels_list, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)

    plt.show()

