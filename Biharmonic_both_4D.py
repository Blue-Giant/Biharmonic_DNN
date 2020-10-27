"""
@author: LXA
Benchmark Code of Biharmonic equations.

"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import DNN_base
import Biharmonic_eqs
import DNN_tools
import DNN_data
import plotData
import saveData
import matData2Biharmonic


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation type for problem: %s\n' % (R_dic['eqs_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(R_dic['act_name']), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)

    if 1 == R['Dirichlet_boundary']:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Dirichlet boundary'), log_fileout)
    else:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Navier boundary'), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_bd_penalty']), log_fileout)
    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_bd, penalty_wb, loss_it_tmp, loss_bd_tmp, loss_bd2_tmp,
                        loss_tmp, train_mse_tmp, train_res_tmp, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('boundary penalty: %f' % temp_penalty_bd)
    print('weights and biases with  penalty: %f' % penalty_wb)
    print('loss_it for training: %.10f' % loss_it_tmp)
    print('loss_bd for training: %.10f' % loss_bd_tmp)
    print('loss_bd to derivative for training: %.10f' % loss_bd2_tmp)
    print('total loss for training: %.10f' % loss_tmp)
    print('function mean square error for training: %.10f' % train_mse_tmp)
    print('function residual error for training: %.10f\n' % train_res_tmp)

    DNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    DNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    DNN_tools.log_string('boundary penalty: %f' % temp_penalty_bd, log_out)
    DNN_tools.log_string('weights and biases with  penalty: %f' % penalty_wb, log_out)
    DNN_tools.log_string('loss_it for training: %.10f' % loss_it_tmp, log_out)
    DNN_tools.log_string('loss_bd for training: %.10f' % loss_bd_tmp, log_out)
    DNN_tools.log_string('loss_bd to derivative for training: %.10f' % loss_bd2_tmp, log_out)
    DNN_tools.log_string('total loss for training: %.10f' % loss_tmp, log_out)
    DNN_tools.log_string('function mean square error for training: %.10f' % train_mse_tmp, log_out)
    DNN_tools.log_string('function residual error for training: %.10f\n' % train_res_tmp, log_out)


def print_and_log2test(mse2test, res2test, log_out=None):
    print('mean square error of predict and real for testing: %.10f' % mse2test)
    print('residual error of predict and real for testing: %.10f\n' % res2test)

    DNN_tools.log_string('mean square error of predict and real for testing: %.10f' % mse2test, log_out)
    DNN_tools.log_string('residual error of predict and real for testing: %.10f\n\n' % res2test, log_out)


def solve_Biharmonic4D(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    bd_penalty_init = R['init_bd_penalty']            # Regularization parameter for boundary conditions
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']
    act_func = R['act_name']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # 问题区域，每个方向设置为一样的长度
    region_lb = 0.0
    region_rt = 1.0
    if R['eqs_type'] == 'Biharmonic4D':
        f, u_true = Biharmonic_eqs.get_biharmonic_infos_4D(input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb,
                                                           right_top=region_rt, laplace_name=R['eqs_name'])

    flag = 'WB'
    hidden_layers = R['hidden_layers']
    # Weights, Biases = Biharmonic_DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = Biharmonic_DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag)
    Weights, Biases = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZS_it = tf.placeholder(tf.float32, name='XYZS', shape=[None, input_dim])
            XYZS00 = tf.placeholder(tf.float32, name='XYZS00', shape=[None, input_dim])
            XYZS01 = tf.placeholder(tf.float32, name='XYZS01', shape=[None, input_dim])
            XYZS10 = tf.placeholder(tf.float32, name='XYZS10', shape=[None, input_dim])
            XYZS11 = tf.placeholder(tf.float32, name='XYZS11', shape=[None, input_dim])
            XYZS20 = tf.placeholder(tf.float32, name='XYZS20', shape=[None, input_dim])
            XYZS21 = tf.placeholder(tf.float32, name='XYZS21', shape=[None, input_dim])
            XYZS30 = tf.placeholder(tf.float32, name='XYZS30', shape=[None, input_dim])
            XYZS31 = tf.placeholder(tf.float32, name='XYZS31', shape=[None, input_dim])
            boundary_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if R['model'] == 'PDE_DNN':
                U_NN = DNN_base.PDE_DNN(XYZS_it, Weights, Biases, hidden_layers, activate_name=act_func)
                U00_NN = DNN_base.PDE_DNN(XYZS00, Weights, Biases, hidden_layers, activate_name=act_func)
                U01_NN = DNN_base.PDE_DNN(XYZS01, Weights, Biases, hidden_layers, activate_name=act_func)
                U10_NN = DNN_base.PDE_DNN(XYZS10, Weights, Biases, hidden_layers, activate_name=act_func)
                U11_NN = DNN_base.PDE_DNN(XYZS11, Weights, Biases, hidden_layers, activate_name=act_func)
                U20_NN = DNN_base.PDE_DNN(XYZS20, Weights, Biases, hidden_layers, activate_name=act_func)
                U21_NN = DNN_base.PDE_DNN(XYZS21, Weights, Biases, hidden_layers, activate_name=act_func)
                U30_NN = DNN_base.PDE_DNN(XYZS30, Weights, Biases, hidden_layers, activate_name=act_func)
                U31_NN = DNN_base.PDE_DNN(XYZS31, Weights, Biases, hidden_layers, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_BN':
                U_NN = DNN_base.PDE_DNN_BN(XYZS_it, Weights, Biases, hidden_layers, activate_name=act_func)
                U00_NN = DNN_base.PDE_DNN_BN(XYZS00, Weights, Biases, hidden_layers, activate_name=act_func)
                U01_NN = DNN_base.PDE_DNN_BN(XYZS01, Weights, Biases, hidden_layers, activate_name=act_func)
                U10_NN = DNN_base.PDE_DNN_BN(XYZS10, Weights, Biases, hidden_layers, activate_name=act_func)
                U11_NN = DNN_base.PDE_DNN_BN(XYZS11, Weights, Biases, hidden_layers, activate_name=act_func)
                U20_NN = DNN_base.PDE_DNN_BN(XYZS20, Weights, Biases, hidden_layers, activate_name=act_func)
                U21_NN = DNN_base.PDE_DNN_BN(XYZS21, Weights, Biases, hidden_layers, activate_name=act_func)
                U30_NN = DNN_base.PDE_DNN_BN(XYZS30, Weights, Biases, hidden_layers, activate_name=act_func)
                U31_NN = DNN_base.PDE_DNN_BN(XYZS31, Weights, Biases, hidden_layers, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN = DNN_base.PDE_DNN_scale(XYZS_it, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U00_NN = DNN_base.PDE_DNN_scale(XYZS00, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U01_NN = DNN_base.PDE_DNN_scale(XYZS01, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U10_NN = DNN_base.PDE_DNN_scale(XYZS10, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U11_NN = DNN_base.PDE_DNN_scale(XYZS11, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U20_NN = DNN_base.PDE_DNN_scale(XYZS20, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U21_NN = DNN_base.PDE_DNN_scale(XYZS21, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U30_NN = DNN_base.PDE_DNN_scale(XYZS30, Weights, Biases, hidden_layers, freq, activate_name=act_func)
                U31_NN = DNN_base.PDE_DNN_scale(XYZS31, Weights, Biases, hidden_layers, freq, activate_name=act_func)
            X_it = tf.reshape(XYZS_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZS_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZS_it[:, 2], shape=[-1, 1])
            S_it = tf.reshape(XYZS_it[:, 3], shape=[-1, 1])

            dU_NN = tf.gradients(U_NN, XYZS_it)[0]
            # 变分形式的loss of interior
            if R['variational_loss'] == 1:
                dU_NN_x1 = tf.gather(dU_NN, [0], axis=-1)
                dU_NN_x2 = tf.gather(dU_NN, [1], axis=-1)
                dU_NN_x3 = tf.gather(dU_NN, [2], axis=-1)
                dU_NN_x4 = tf.gather(dU_NN, [3], axis=-1)

                ddU_NN_x1 = tf.gradients(dU_NN_x1, XYZS_it)[0]
                ddU_x1 = tf.gather(ddU_NN_x1, [0], axis=-1)

                ddU_NN_x2 = tf.gradients(dU_NN_x2, XYZS_it)[0]
                ddU_x2 = tf.gather(ddU_NN_x2, [1], axis=-1)

                ddU_NN_x3 = tf.gradients(dU_NN_x3, XYZS_it)[0]
                ddU_x3 = tf.gather(ddU_NN_x3, [2], axis=-1)

                ddU_NN_x4 = tf.gradients(dU_NN_x4, XYZS_it)[0]
                ddU_x4 = tf.gather(ddU_NN_x4, [3], axis=-1)

                # ddU_x1x2 = tf.gather(ddU_NN_x1, [1], axis=-1)
                # ddU_x1x3 = tf.gather(ddU_NN_x1, [2], axis=-1)
                # ddU_x2x3 = tf.gather(ddU_NN_x2, [2], axis=-1)

                laplace_norm = tf.square(ddU_x1) + tf.square(ddU_x2) + tf.square(ddU_x3) + tf.square(ddU_x4) + \
                               2.0 * tf.multiply(ddU_x1, ddU_x2) + 2.0 * tf.multiply(ddU_x1, ddU_x3) + \
                               2.0 * tf.multiply(ddU_x1, ddU_x4) + 2.0 * tf.multiply(ddU_x2, ddU_x3) + \
                               2.0 * tf.multiply(ddU_x2, ddU_x4) + 2.0 * tf.multiply(ddU_x3, ddU_x4)
                loss_it_variational = 0.5 * laplace_norm - f(X_it, Y_it, Z_it, S_it) * U_NN
                Loss_it = tf.reduce_mean(loss_it_variational) * np.power(region_rt - region_lb, input_dim)

                # 边界loss，首先利用训练集把准确的边界值得到，然后和 neural  network 训练结果作差，最后平方
                loss_bd_square = tf.square(U00_NN) + tf.square(U01_NN) + tf.square(U10_NN) + tf.square(U11_NN) + \
                                 tf.square(U20_NN) + tf.square(U21_NN) + tf.square(U30_NN) + tf.square(U31_NN)
                Loss_bd = tf.reduce_mean(loss_bd_square)

            # 边界上的偏导数(一阶偏导和二阶偏导)
            if R['Dirichlet_boundary'] == 1:
                dU00_NN_temp = tf.gradients(U00_NN, XYZS00)[0]
                dU00_NN = -1.0 * tf.gather(dU00_NN_temp, [0], axis=-1)

                dU01_NN_temp = tf.gradients(U01_NN, XYZS01)[0]
                dU01_NN = tf.gather(dU01_NN_temp, [0], axis=-1)

                dU10_NN_temp = tf.gradients(U10_NN, XYZS10)[0]
                dU10_NN = -1.0 * tf.gather(dU10_NN_temp, [1], axis=-1)

                dU11_NN_temp = tf.gradients(U11_NN, XYZS11)[0]
                dU11_NN = tf.gather(dU11_NN_temp, [1], axis=-1)

                dU20_NN_temp = tf.gradients(U20_NN, XYZS20)[0]
                dU20_NN = -1.0 * tf.gather(dU20_NN_temp, [2], axis=-1)

                dU21_NN_temp = tf.gradients(U21_NN, XYZS21)[0]
                dU21_NN = tf.gather(dU21_NN_temp, [2], axis=-1)

                dU30_NN_temp = tf.gradients(U30_NN, XYZS30)[0]
                dU30_NN = -1.0 * tf.gather(dU30_NN_temp, [3], axis=-1)

                dU31_NN_temp = tf.gradients(U31_NN, XYZS31)[0]
                dU31_NN = tf.gather(dU31_NN_temp, [3], axis=-1)

                loss_1derivative = tf.square(dU00_NN) + tf.square(dU01_NN) + tf.square(dU10_NN) + tf.square(dU11_NN) + \
                                   tf.square(dU20_NN) + tf.square(dU21_NN) + tf.square(dU30_NN) + tf.square(dU31_NN)
                Loss_bdd = tf.reduce_mean(loss_1derivative)
            if R['Navier_boundary'] == 1:
                dU00_NN_temp = tf.gradients(U00_NN, XYZS00)[0]
                dU00_NN_x = tf.gather(dU00_NN_temp, [0], axis=-1)
                dU00_NN_y = tf.gather(dU00_NN_temp, [1], axis=-1)
                dU00_NN_z = tf.gather(dU00_NN_temp, [2], axis=-1)
                dU00_NN_s = tf.gather(dU00_NN_temp, [3], axis=-1)
                ddU00_NN_x = tf.gradients(dU00_NN_x, XYZS00)[0]
                U00_NN_xx = tf.gather(ddU00_NN_x, [0], axis=-1)
                ddU00_NN_y = tf.gradients(dU00_NN_y, XYZS00)[0]
                U00_NN_yy = tf.gather(ddU00_NN_y, [1], axis=-1)
                ddU00_NN_z = tf.gradients(dU00_NN_z, XYZS00)[0]
                U00_NN_zz = tf.gather(ddU00_NN_z, [2], axis=-1)
                ddU00_NN_s = tf.gradients(dU00_NN_s, XYZS00)[0]
                U00_NN_ss = tf.gather(ddU00_NN_s, [3], axis=-1)
                laplaceU00_NN = U00_NN_xx + U00_NN_yy + U00_NN_zz + U00_NN_ss

                dU01_NN_temp = tf.gradients(U01_NN, XYZS01)[0]
                dU01_NN_x = tf.gather(dU01_NN_temp, [0], axis=-1)
                dU01_NN_y = tf.gather(dU01_NN_temp, [1], axis=-1)
                dU01_NN_z = tf.gather(dU01_NN_temp, [2], axis=-1)
                dU01_NN_s = tf.gather(dU01_NN_temp, [3], axis=-1)
                ddU01_NN_x = tf.gradients(dU01_NN_x, XYZS01)[0]
                U01_NN_xx = tf.gather(ddU01_NN_x, [0], axis=-1)
                ddU01_NN_y = tf.gradients(dU01_NN_y, XYZS01)[0]
                U01_NN_yy = tf.gather(ddU01_NN_y, [1], axis=-1)
                ddU01_NN_z = tf.gradients(dU01_NN_z, XYZS01)[0]
                U01_NN_zz = tf.gather(ddU01_NN_z, [2], axis=-1)
                ddU01_NN_s = tf.gradients(dU01_NN_s, XYZS01)[0]
                U01_NN_ss = tf.gather(ddU01_NN_s, [3], axis=-1)
                laplaceU01_NN = U01_NN_xx + U01_NN_yy + U01_NN_zz + U01_NN_ss

                dU10_NN_temp = tf.gradients(U10_NN, XYZS10)[0]
                dU10_NN_x = tf.gather(dU10_NN_temp, [0], axis=-1)
                dU10_NN_y = tf.gather(dU10_NN_temp, [1], axis=-1)
                dU10_NN_z = tf.gather(dU10_NN_temp, [2], axis=-1)
                dU10_NN_s = tf.gather(dU10_NN_temp, [3], axis=-1)
                ddU10_NN_x = tf.gradients(dU10_NN_x, XYZS10)[0]
                U10_NN_xx = tf.gather(ddU10_NN_x, [0], axis=-1)
                ddU10_NN_y = tf.gradients(dU10_NN_y, XYZS10)[0]
                U10_NN_yy = tf.gather(ddU10_NN_y, [1], axis=-1)
                ddU10_NN_z = tf.gradients(dU10_NN_z, XYZS10)[0]
                U10_NN_zz = tf.gather(ddU10_NN_z, [2], axis=-1)
                ddU10_NN_s = tf.gradients(dU10_NN_s, XYZS10)[0]
                U10_NN_ss = tf.gather(ddU10_NN_s, [3], axis=-1)
                laplaceU10_NN = U10_NN_xx + U10_NN_yy + U10_NN_zz + U10_NN_ss

                dU11_NN_temp = tf.gradients(U11_NN, XYZS11)[0]
                dU11_NN_x = tf.gather(dU11_NN_temp, [0], axis=-1)
                dU11_NN_y = tf.gather(dU11_NN_temp, [1], axis=-1)
                dU11_NN_z = tf.gather(dU11_NN_temp, [2], axis=-1)
                dU11_NN_s = tf.gather(dU11_NN_temp, [3], axis=-1)
                ddU11_NN_x = tf.gradients(dU11_NN_x, XYZS11)[0]
                U11_NN_xx = tf.gather(ddU11_NN_x, [0], axis=-1)
                ddU11_NN_y = tf.gradients(dU11_NN_y, XYZS11)[0]
                U11_NN_yy = tf.gather(ddU11_NN_y, [1], axis=-1)
                ddU11_NN_z = tf.gradients(dU11_NN_z, XYZS11)[0]
                U11_NN_zz = tf.gather(ddU11_NN_z, [2], axis=-1)
                ddU11_NN_s = tf.gradients(dU11_NN_s, XYZS11)[0]
                U11_NN_ss = tf.gather(ddU11_NN_s, [3], axis=-1)
                laplaceU11_NN = U11_NN_xx + U11_NN_yy + U11_NN_zz + U11_NN_ss

                dU20_NN_temp = tf.gradients(U20_NN, XYZS20)[0]
                dU20_NN_x = tf.gather(dU20_NN_temp, [0], axis=-1)
                dU20_NN_y = tf.gather(dU20_NN_temp, [1], axis=-1)
                dU20_NN_z = tf.gather(dU20_NN_temp, [2], axis=-1)
                dU20_NN_s = tf.gather(dU20_NN_temp, [3], axis=-1)
                ddU20_NN_x = tf.gradients(dU20_NN_x, XYZS20)[0]
                U20_NN_xx = tf.gather(ddU20_NN_x, [0], axis=-1)
                ddU20_NN_y = tf.gradients(dU20_NN_y, XYZS20)[0]
                U20_NN_yy = tf.gather(ddU20_NN_y, [1], axis=-1)
                ddU20_NN_z = tf.gradients(dU20_NN_z, XYZS20)[0]
                U20_NN_zz = tf.gather(ddU20_NN_z, [2], axis=-1)
                ddU20_NN_s = tf.gradients(dU20_NN_s, XYZS20)[0]
                U20_NN_ss = tf.gather(ddU20_NN_s, [3], axis=-1)
                laplaceU20_NN = U20_NN_xx + U20_NN_yy + U20_NN_zz + U20_NN_ss

                dU21_NN_temp = tf.gradients(U21_NN, XYZS21)[0]
                dU21_NN_x = tf.gather(dU21_NN_temp, [0], axis=-1)
                dU21_NN_y = tf.gather(dU21_NN_temp, [1], axis=-1)
                dU21_NN_z = tf.gather(dU21_NN_temp, [2], axis=-1)
                dU21_NN_s = tf.gather(dU21_NN_temp, [3], axis=-1)
                ddU21_NN_x = tf.gradients(dU21_NN_x, XYZS21)[0]
                U21_NN_xx = tf.gather(ddU21_NN_x, [0], axis=-1)
                ddU21_NN_y = tf.gradients(dU21_NN_y, XYZS21)[0]
                U21_NN_yy = tf.gather(ddU21_NN_y, [1], axis=-1)
                ddU21_NN_z = tf.gradients(dU21_NN_z, XYZS21)[0]
                U21_NN_zz = tf.gather(ddU21_NN_z, [2], axis=-1)
                ddU21_NN_s = tf.gradients(dU21_NN_s, XYZS21)[0]
                U21_NN_ss = tf.gather(ddU21_NN_s, [3], axis=-1)
                laplaceU21_NN = U21_NN_xx + U21_NN_yy + U21_NN_zz + U21_NN_ss

                dU30_NN_temp = tf.gradients(U30_NN, XYZS30)[0]
                dU30_NN_x = tf.gather(dU30_NN_temp, [0], axis=-1)
                dU30_NN_y = tf.gather(dU30_NN_temp, [1], axis=-1)
                dU30_NN_z = tf.gather(dU30_NN_temp, [2], axis=-1)
                dU30_NN_s = tf.gather(dU30_NN_temp, [3], axis=-1)
                ddU30_NN_x = tf.gradients(dU30_NN_x, XYZS30)[0]
                U30_NN_xx = tf.gather(ddU30_NN_x, [0], axis=-1)
                ddU30_NN_y = tf.gradients(dU30_NN_y, XYZS30)[0]
                U30_NN_yy = tf.gather(ddU30_NN_y, [1], axis=-1)
                ddU30_NN_z = tf.gradients(dU30_NN_z, XYZS30)[0]
                U30_NN_zz = tf.gather(ddU30_NN_z, [2], axis=-1)
                ddU30_NN_s = tf.gradients(dU30_NN_s, XYZS30)[0]
                U30_NN_ss = tf.gather(ddU30_NN_s, [3], axis=-1)
                laplaceU30_NN = U30_NN_xx + U30_NN_yy + U30_NN_zz + U30_NN_ss

                dU31_NN_temp = tf.gradients(U31_NN, XYZS31)[0]
                dU31_NN_x = tf.gather(dU31_NN_temp, [0], axis=-1)
                dU31_NN_y = tf.gather(dU31_NN_temp, [1], axis=-1)
                dU31_NN_z = tf.gather(dU31_NN_temp, [2], axis=-1)
                dU31_NN_s = tf.gather(dU31_NN_temp, [3], axis=-1)
                ddU31_NN_x = tf.gradients(dU31_NN_x, XYZS31)[0]
                U31_NN_xx = tf.gather(ddU31_NN_x, [0], axis=-1)
                ddU31_NN_y = tf.gradients(dU31_NN_y, XYZS31)[0]
                U31_NN_yy = tf.gather(ddU31_NN_y, [1], axis=-1)
                ddU31_NN_z = tf.gradients(dU31_NN_z, XYZS31)[0]
                U31_NN_zz = tf.gather(ddU31_NN_z, [2], axis=-1)
                ddU31_NN_s = tf.gradients(dU31_NN_s, XYZS31)[0]
                U31_NN_ss = tf.gather(ddU31_NN_s, [3], axis=-1)
                laplaceU31_NN = U31_NN_xx + U31_NN_yy + U31_NN_zz + U31_NN_ss

                loss_2derivative = tf.square(laplaceU00_NN) + tf.square(laplaceU01_NN) + tf.square(laplaceU10_NN) + \
                                   tf.square(laplaceU11_NN)+ tf.square(laplaceU20_NN) + tf.square(laplaceU21_NN) + \
                                   tf.square(laplaceU30_NN) + tf.square(laplaceU31_NN)
                Loss_bdd = tf.reduce_mean(loss_2derivative)

            if R['regular_weight_model'] == 'L1':
                regular_WB = DNN_base.regular_weights_biases_L1(Weights, Biases)      # 正则化权重参数 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = DNN_base.regular_weights_biases_L2(Weights, Biases)      # 正则化权重参数 L2正则化
            else:
                regular_WB = tf.constant(0.0)                                                                 # 无正则化权重参数

            PWB = wb_penalty * regular_WB
            Loss = Loss_it + boundary_penalty * Loss_bd + boundary_penalty * Loss_bdd  # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss = my_optimizer.minimize(Loss, global_step=global_steps)

            U_true = u_true(X_it, Y_it, Z_it, S_it)
            train_Mse = tf.reduce_mean(tf.square(U_true - U_NN))
            train_Rel = train_Mse/tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_bdd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], [], []
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        test_xyzs_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyzs_bach, dataName='testXYZS', outPath=R['FolderName'])
    else:
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'data2mat'
        test_xyzs_bach = matData2Biharmonic.get_data2Biharmonic(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyzs_bach, dataName='testXYZS', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            XYZS_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            XYZS00_batch, XYZS01_batch, XYZS10_batch, XYZS11_batch, XYZS20_batch, XYZS21_batch, XYZS30_batch, \
            XYZS31_batch = DNN_data.rand_bd_4D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < 2*int(R['max_epoch'] / 3):
                    temp_penalty_bd = 10 * bd_penalty_init
                else:
                    temp_penalty_bd = 50 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

                _, loss_it_tmp, loss_bd_tmp, loss_bdd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_Loss, Loss_it, Loss_bd, Loss_bdd, Loss, train_Mse, train_Rel, PWB],
                feed_dict={XYZS_it: XYZS_it_batch, XYZS00: XYZS00_batch, XYZS01: XYZS01_batch, XYZS10: XYZS10_batch,
                           XYZS11: XYZS11_batch, XYZS20: XYZS20_batch, XYZS21: XYZS21_batch, XYZS30: XYZS30_batch,
                           XYZS31: XYZS31_batch, in_learning_rate: tmp_lr, train_opt: train_option,
                           boundary_penalty: temp_penalty_bd})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            loss_bdd_all.append(loss_bdd_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp,
                                    loss_bdd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_nn2test = sess.run(
                    [U_true, U_NN], feed_dict={XYZS_it: test_xyzs_bach, train_opt: train_option})
                point_square_error = np.square(u_true2test - u_nn2test)
                mse2test = np.mean(point_square_error)
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)
                print('mean square error of predict and real for testing: %10f' % mse2test)
                print('residual error of predict and real for testing: %10f\n' % res2test)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % mse2test, log_fileout)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % res2test, log_fileout)

            if (i_epoch != 0 and i_epoch != 100000) and i_epoch % 10000 == 0 and R['Navier_boundary'] == 1:
                pathOut = '%s/%s' % (R['FolderName'], int(i_epoch / 10000))
                print('------- i_epoch-------:', i_epoch)
                print('\n')
                print('------- pathOut-------:', pathOut)
                if not os.path.exists(pathOut):  # 判断路径是否已经存在
                    os.mkdir(pathOut)  # 无 log_out_path 路径，创建一个 log_out_path 路径

                saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=pathOut)
                saveData.save_testData_or_solus2mat(u_nn2test, dataName=act_func, outPath=pathOut)
                # 绘解得热力图
                if R['hot_power'] == 1:
                    # 绘解得热力图
                    plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                                    seedNo=R['seed'], outPath=R['FolderName'])
                    # 绘制预测解得热力图
                    plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName='s2ReLU',
                                                    seedNo=R['seed'], outPath=R['FolderName'])

                saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=pathOut)
                plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                          seedNo=R['seed'], outPath=pathOut, yaxis_scale=True)

                # 保存绘制误差的能量图
                saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=pathOut)
                plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
                                                 seedNo=R['seed'], outPath=pathOut)

        # -----------------------  save training result to mat file, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                             outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'],
                                          outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'],
                                          outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ------------------------------ save testing result to mat file, then plot them -------------------------------
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(u_nn2test, dataName=act_func, outPath=R['FolderName'])
        if R['hot_power'] == 1:
            # 绘解得热力图
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                            seedNo=R['seed'], outPath=R['FolderName'])
            # 绘制预测解得热力图
            plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName=act_func,
                                            seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName='s2ReLU', outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # 绘制误差的能量图
        saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName='s2ReLU',
                                         seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 1  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'pos1'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # ---------------------------- Setup of PDE_DNNs -------------------------------
    R['eqs_type'] = 'Biharmonic4D'
    # R['eqs_name'] = 'Dirichlet_equation'
    R['eqs_name'] = 'Navier_equation'

    R['input_dim'] = 4                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['variational_loss'] = 1             # PDE变分
    if R['eqs_name'] == 'Navier_equation':
        R['Dirichlet_boundary'] = 0
        R['Navier_boundary'] = 1
    else:
        R['Dirichlet_boundary'] = 1
        R['Navier_boundary'] = 0

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['hot_power'] = 0
    R['batch_size2interior'] = 7500        # 内部训练数据的批大小
    R['batch_size2boundary'] = 1250        # 边界训练数据的批大小
    R['testData_model'] = 'loadData'

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L2'

    R['init_bd_penalty'] = 500           # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_bd_penalty'] = 5

    R['regular_weight'] = 0.000           # Regularization parameter for weights
    # R['regular_weight'] = 0.001         # Regularization parameter for weights
    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    elif 20000 < R['max_epoch'] and 50000 > R['max_epoch']:
        R['learning_rate'] = 1e-4  # 学习率
        R['lr_decay'] = 4e-5       # 学习率 decay
    else:
        R['learning_rate'] = 5e-5  # 学习率
        R['lr_decay'] = 1e-5       # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器

    if R['eqs_name'] == 'Dirichlet_equation':
        R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
    elif R['eqs_name'] == 'Navier_equation':
        R['hidden_layers'] = (300, 300, 200, 200, 100, 100)
        # R['hidden_layers'] = (200, 150, 150, 100, 80, 80)
        # R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
    else:
        R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
        # R['hidden_layers'] = (300, 200, 200, 100, 80, 80, 50)
        # R['hidden_layers'] = (400, 300, 300, 200, 100, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100, 100)
        # R['hidden_layers'] = (600, 400, 400, 300, 200, 200, 100)
        # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)

    # 网络模型的选择
    R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scale'

    # 激活函数的选择
    # R['act_name'] = 'relu'
    # R['act_name'] = 'tanh'
    # R['act_name']' = leaky_relu'
    # R['act_name'] = 'srelu'
    R['act_name'] = 'sin_srelu'
    # R['act_name'] = 'slrelu'
    # R['act_name'] = 'elu'
    # R['act_name'] = 'selu'
    # R['act_name'] = 'phi'

    solve_Biharmonic4D(R)
