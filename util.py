from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import math
from scipy import interpolate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import csv

from pdb import set_trace as bp

def green_print(line):
    print('\033[92m'+line+'\033[0m')

def create_output_path(csv_filepath):
    # Create folder and filename
    if not os.path.exists('result'):
        os.mkdir('result')
    filepath = csv_filepath.split('/')[-1]
    filepath = 'result/Simulator_v4_%s' % filepath.split('.')[-2]
    if os.path.exists(filepath):
        os.remove(filepath)
    return filepath

def readEmb_csv(filepath):
    # Read labels
    with open(filepath, 'r') as textfile:
        reader =  csv.DictReader(textfile, ["names", "features", "thresholds", "filepaths"])
        all_data = [[row["names"], row["features"]] for row in reader]

    # Transfer data
    embeddings = []
    label_array = []
    for data in all_data:
        tmp = data[1].replace('[', '')
        tmp = tmp.replace(']', '')
        tmp = tmp.replace('\n', '')
        embeddings.append(tmp)
        label_array.append(data[0])

    emb_array = np.zeros((len(embeddings), 128), dtype=float)
    for indx, emb in enumerate(embeddings):
        emb_array[indx, :] = np.fromstring(emb, dtype=np.float32, sep=" ")
    return emb_array, label_array


def get_batch(emb_array, labels, testing_index):
    test_array = np.expand_dims(emb_array[testing_index], axis=0)
    test_label = [labels[testing_index]]
    return test_array, test_label

def calculate_error(fa, fr, wa, accept, reject):
    if accept > 0:
        far = float(fa/accept)
        war = float(wa/accept)
    else:
        far = 0
        war = 0

    if reject > 0:
        frr = float(fr/reject)
    else:
        frr = 0
    
    error = (fa+fr+wa)/(accept+reject)
    return far, frr, war, error

def show_result(cur_thd, fa, fr, wa, accept, reject):
    far, frr, war,error = calculate_error(fa, fr, wa, accept, reject)
    similarity = float((2-cur_thd**2)/2)
    green_print('Threshold = %.4f: far:%f(%d/%d), frr:%f(%d/%d), war:%f(%d/%d), total:%.4f(%d/%d)' % \
               (similarity, far, fa, accept, frr, fr, reject, 
                war, wa, accept, error, (fa+fr+wa), (accept+reject)))

def show_and_save_v3(fa, fr, wa, accept, reject, compare_num, filepath):
    # Calculate error
    far, frr, war, error = calculate_error(fa, fr, wa, accept, reject)
    info = 'compare_num: %d\nfar:%f(%d/%d), frr:%f(%d/%d), war:%f(%d/%d), acc:%.4f(%d/%d)\n' % \
            (compare_num, far, fa, accept, frr, fr, reject, 
            war, wa, accept, 1-error, (fa+fr+wa), (accept+reject))
    # Print result
    green_print(info)
    # Save result
    with open(filepath, 'a') as file:
        file.write(info)
    return filepath

def get_rate(key, index, rates):
    rate = [rates[i][index] for i in range(len(rates))]
    rate = [v.split(key)[-1] for v in rate]
    rate = [v.split('(')[0] for v in rate]
    rate = [float(v) for v in rate]
    return rate

def findIntersection(fun1, fun2, x0):
 return fsolve(lambda x : fun1(x) - fun2(x), x0)

def plot(result_file, start):
    # Read file
    with open(result_file, 'r') as file:
        all_data = file.read().split('\n')
    if all_data[-1] == '':
        del all_data[-1]

    # Split x and y data
    compare_num = []
    rates = []
    for indx, v in enumerate(all_data):
        if indx%2 == 0:
            compare_num.append(float(v.split(': ')[-1]))
        else:
            rates.append(v.split(', '))

    # Get y axis data
    far = get_rate('far:', 0, rates)
    frr = get_rate('frr:', 1, rates)
    war = get_rate('war:', 2, rates)
    acc = get_rate('acc:', 3, rates)

    # Find max accuracy
    max_acc_index = np.argmax(acc)
    best_acc = acc[max_acc_index]
    best_num = compare_num[max_acc_index]
    best_far = far[max_acc_index]
    best_frr = frr[max_acc_index]

    # Find EER point
    f1 = interpolate.interp1d(compare_num, far)
    f2 = interpolate.interp1d(compare_num, frr)
    f3 = interpolate.interp1d(compare_num, acc)

    compare_num_EER = findIntersection(f1, f2, start)

    # Plot begin
    plt.figure(figsize=(14,6))

    # Error rates
    plt.subplot(121)
    plt.plot(compare_num, far, label='FAR')
    plt.plot(compare_num, frr, label='FRR')
    plt.plot(compare_num, war, label='WAR')
    
    # Points
    EER_label = ('EER: %.4f/%d' % (f1(compare_num_EER), compare_num_EER))
    acc_far_label = ('FAR: %.4f/%d' % (best_far, best_num))
    acc_frr_label = ('FRR: %.4f/%d' % (best_frr, best_num))
    plt.plot(compare_num_EER, f1(compare_num_EER), 'ro', label=EER_label)
    plt.plot(best_num, best_far, 'go', label=acc_far_label)
    plt.plot(best_num, best_frr, 'bo', label=acc_frr_label)
    
    plt.xlabel('Max number of class compared')
    plt.ylabel('Error rate')
    plt.title('Adaptive threshold - Error rates')
    plt.legend(loc=4)
    
    # Accuracy
    plt.subplot(122)
    plt.plot(compare_num, acc)
    max_label = ('Best: %.4f/%d' % (best_acc, best_num))
    acc_EER_label = ('EER: %.4f/%d' % (f3(compare_num_EER), compare_num_EER))
    plt.plot(best_num, best_acc, 'ro', label=max_label)
    plt.plot(compare_num_EER, f3(compare_num_EER), 'bo', label=acc_EER_label)
    plt.xlabel('Max number of class compared')
    plt.ylabel('Accuracy')
    plt.title('Adaptive threshold - Accuracy')
    plt.legend()


    plt.show()
