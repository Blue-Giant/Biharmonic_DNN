"""
@author: LXA
Benchmark Code of Laplace equations.

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
import saveData
import plotData
import DNN_boundary


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName=None):
    DNN_tools.log_string('Equation type for problem: %s\n' % (R_dic['eqs_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(actName), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if 1 == R['Navier_boundary']:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Navier boundary'), log_fileout)
    else:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Dirichlet boundary'), log_fileout)
    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_bd_penalty']), log_fileout)
    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

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


def solve_Biharmonic3D(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径

    outfile_name1 = '%s_%s.txt' % ('log2train', R['act_name2NN1'])
    log_fileout_NN1 = open(os.path.join(log_out_path, outfile_name1), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout_NN1, actName=R['act_name2NN1'])

    outfile_name = '%s_%s.txt' % ('log2train', R['act_name2NN2'])
    log_fileout_NN2 = open(os.path.join(log_out_path, outfile_name), 'w')
    dictionary_out2file(R, log_fileout_NN2, actName=R['act_name2NN2'])

    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    bd_penalty_init = R['init_bd_penalty']            # Regularization parameter for boundary conditions
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # 问题区域，每个方向设置为一样的长度
    region_lb = 0.0
    region_rt = 1.0
    if R['eqs_type'] == 'Biharmonic_3D':
        # laplace laplace u = f
        f, u_true, u_bottom, u_top, u_left, u_right, u_front, u_behind = Biharmonic_eqs.get_biharmonic_Navier_3D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, laplace_name=R['eqs_name'])

    flag2u_NN1 = 'WB2u_NN1'
    flag2psi_NN1 = 'WB2psi_NN1'
    flag2u_NN2 = 'WB2u_NN2'
    flag2psi_NN2 = 'WB2psi_NN2'
    hidden_layers = R['hidden_layers']
    # Weights, Biases = ThinPlate_DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = ThinPlate_DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag)
    Weights2u_NN1, Bias2u_NN1 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2u_NN1)
    Weights2psi_NN1, Bias2psi_NN1 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers,
                                                                          flag2psi_NN1)
    Weights2u_NN2, Bias2u_NN2 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2u_NN2)
    Weights2psi_NN2, Bias2psi_NN2 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers,
                                                                          flag2psi_NN2)

    # Laplace Laplace u = f
    # Navier case: u = Laplace u=0
    # Dirichlet case: u = Bu/Bn=0
    # Laplace u = -psi <--> -Laplace u = psi
    # 那么我们有   - Laplace psi=f
    #              - Laplace u = psi
    #              u = partial u/partial n=0
    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZ_it = tf.placeholder(tf.float32, name='XYZ_it', shape=[None, input_dim])
            # XYZ_it = tf.placeholder(tf.float32, name='XYZ_it', shape=[batchsize_it, input_dim])
            XYZ_bottom_bd = tf.placeholder(tf.float32, name='bottom_bd', shape=[None, input_dim])
            XYZ_top_bd = tf.placeholder(tf.float32, name='top_bd', shape=[None, input_dim])
            XYZ_left_bd = tf.placeholder(tf.float32, name='left_bd', shape=[None, input_dim])
            XYZ_right_bd = tf.placeholder(tf.float32, name='right_bd', shape=[None, input_dim])
            XYZ_front_bd = tf.placeholder(tf.float32, name='front_bd', shape=[None, input_dim])
            XYZ_behind_bd = tf.placeholder(tf.float32, name='behind_bd', shape=[None, input_dim])
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if 'PDE_DNN' == str.upper(R['model']):
                Psi_NN1 = DNN_base.PDE_DNN(XYZ_it, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiBottom_NN1 = DNN_base.PDE_DNN(XYZ_bottom_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiTop_NN1 = DNN_base.PDE_DNN(XYZ_top_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiLeft_NN1 = DNN_base.PDE_DNN(XYZ_left_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiRight_NN1 = DNN_base.PDE_DNN(XYZ_right_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiFront_NN1 = DNN_base.PDE_DNN(XYZ_front_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)
                PsiBehind_NN1 = DNN_base.PDE_DNN(XYZ_behind_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, activate_name=act_func1)

                U_NN1 = DNN_base.PDE_DNN(XYZ_it, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN(XYZ_bottom_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN(XYZ_top_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN(XYZ_left_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN(XYZ_right_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                UFront_NN1 = DNN_base.PDE_DNN(XYZ_front_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)
                UBehind_NN1 = DNN_base.PDE_DNN(XYZ_behind_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, activate_name=act_func1)

                Psi_NN2 = DNN_base.PDE_DNN(XYZ_it, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, activate_name=act_func2)
                PsiBottom_NN2 = DNN_base.PDE_DNN(XYZ_bottom_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                 activate_name=act_func2)
                PsiTop_NN2 = DNN_base.PDE_DNN(XYZ_top_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                              activate_name=act_func2)
                PsiLeft_NN2 = DNN_base.PDE_DNN(XYZ_left_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                               activate_name=act_func2)
                PsiRight_NN2 = DNN_base.PDE_DNN(XYZ_right_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                activate_name=act_func2)
                PsiFront_NN2 = DNN_base.PDE_DNN(XYZ_front_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                activate_name=act_func2)
                PsiBehind_NN2 = DNN_base.PDE_DNN(XYZ_behind_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                 activate_name=act_func2)

                U_NN2 = DNN_base.PDE_DNN(XYZ_it, Weights2u_NN2, Bias2u_NN2, hidden_layers, activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN(XYZ_bottom_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                               activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN(XYZ_top_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                            activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN(XYZ_left_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                             activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN(XYZ_right_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                              activate_name=act_func2)
                UFront_NN2 = DNN_base.PDE_DNN(XYZ_front_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                              activate_name=act_func2)
                UBehind_NN2 = DNN_base.PDE_DNN(XYZ_behind_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers,
                                               activate_name=act_func2)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                Psi_NN1 = DNN_base.PDE_DNN_scale(XYZ_it, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                 activate_name=act_func1)
                PsiBottom_NN1 = DNN_base.PDE_DNN_scale(XYZ_bottom_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                       activate_name=act_func1)
                PsiTop_NN1 = DNN_base.PDE_DNN_scale(XYZ_top_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                    activate_name=act_func1)
                PsiLeft_NN1 = DNN_base.PDE_DNN_scale(XYZ_left_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                     activate_name=act_func1)
                PsiRight_NN1 = DNN_base.PDE_DNN_scale(XYZ_right_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                      activate_name=act_func1)
                PsiFront_NN1 = DNN_base.PDE_DNN_scale(XYZ_front_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                      activate_name=act_func1)
                PsiBehind_NN1 = DNN_base.PDE_DNN_scale(XYZ_behind_bd, Weights2psi_NN1, Bias2psi_NN1, hidden_layers, freq,
                                                       activate_name=act_func1)

                U_NN1 = DNN_base.PDE_DNN_scale(XYZ_it, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                               activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN_scale(XYZ_bottom_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                     activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN_scale(XYZ_top_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                  activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_scale(XYZ_left_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                   activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_scale(XYZ_right_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                    activate_name=act_func1)
                UFront_NN1 = DNN_base.PDE_DNN_scale(XYZ_front_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                    activate_name=act_func1)
                UBehind_NN1 = DNN_base.PDE_DNN_scale(XYZ_behind_bd, Weights2u_NN1, Bias2u_NN1, hidden_layers, freq,
                                                     activate_name=act_func1)

                Psi_NN2 = DNN_base.PDE_DNN_scale(XYZ_it, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, freq,
                                                 activate_name=act_func2)
                PsiBottom_NN2 = DNN_base.PDE_DNN_scale(XYZ_bottom_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                       freq, activate_name=act_func2)
                PsiTop_NN2 = DNN_base.PDE_DNN_scale(XYZ_top_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, freq,
                                                    activate_name=act_func2)
                PsiLeft_NN2 = DNN_base.PDE_DNN_scale(XYZ_left_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, freq,
                                                     activate_name=act_func2)
                PsiRight_NN2 = DNN_base.PDE_DNN_scale(XYZ_right_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, freq,
                                                      activate_name=act_func2)
                PsiFront_NN2 = DNN_base.PDE_DNN_scale(XYZ_front_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers, freq,
                                                      activate_name=act_func2)
                PsiBehind_NN2 = DNN_base.PDE_DNN_scale(XYZ_behind_bd, Weights2psi_NN2, Bias2psi_NN2, hidden_layers,
                                                       freq, activate_name=act_func2)

                U_NN2 = DNN_base.PDE_DNN_scale(XYZ_it, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                               activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN_scale(XYZ_bottom_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                     activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN_scale(XYZ_top_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                  activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_scale(XYZ_left_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                   activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_scale(XYZ_right_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                    activate_name=act_func2)
                UFront_NN2 = DNN_base.PDE_DNN_scale(XYZ_front_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                    activate_name=act_func2)
                UBehind_NN2 = DNN_base.PDE_DNN_scale(XYZ_behind_bd, Weights2u_NN2, Bias2u_NN2, hidden_layers, freq,
                                                     activate_name=act_func2)

            X_it = tf.reshape(XYZ_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZ_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZ_it[:, 2], shape=[-1, 1])
            dPsi_NN1 = tf.gradients(Psi_NN1, XYZ_it)[0]
            dU_NN1 = tf.gradients(U_NN1, XYZ_it)[0]

            dPsi_NN2 = tf.gradients(Psi_NN2, XYZ_it)[0]
            dU_NN2 = tf.gradients(U_NN2, XYZ_it)[0]
            if R['variational_loss'] != 0:
                fff = f(X_it, Y_it, Z_it)
                uf = tf.multiply(Psi_NN1, f(X_it, Y_it, Z_it))
                laplace2Psi_NN1 = tf.reshape(tf.reduce_sum(tf.square(dPsi_NN1), axis=-1), shape=[-1, 1])
                laplace2U_NN1 = tf.reshape(tf.reduce_sum(tf.square(dU_NN1), axis=-1), shape=[-1, 1])
                loss_it2Psi_NN1 = tf.reduce_mean(0.5 * laplace2Psi_NN1 - tf.multiply(Psi_NN1, f(X_it, Y_it, Z_it)))
                loss_it2U_NN1 = tf.reduce_mean(0.5 * laplace2U_NN1 - tf.multiply(Psi_NN1, U_NN1))

                laplace2Psi_NN2 = tf.reshape(tf.reduce_sum(tf.square(dPsi_NN2), axis=-1), shape=[-1, 1])
                laplace2U_NN2 = tf.reshape(tf.reduce_sum(tf.square(dU_NN2), axis=-1), shape=[-1, 1])
                loss_it2Psi_NN2 = tf.reduce_mean(0.5 * laplace2Psi_NN2 - tf.multiply(Psi_NN2, f(X_it, Y_it,Z_it)))
                loss_it2U_NN2 = tf.reduce_mean(0.5 * laplace2U_NN2 - tf.multiply(Psi_NN2, U_NN2))

            # # 边界loss，首先利用训练集把准确的边界值得到，然后和 neural  network 训练结果作差，最后平方
            # ubottom = u_bottom(XYZ_bottom_bd[:, 0], XYZ_bottom_bd[:, 1], XYZ_bottom_bd[:, 2])
            # utop = u_top(XYZ_top_bd[:, 0], XYZ_top_bd[:, 1], XYZ_top_bd[:, 2])
            # uleft = u_left(XYZ_left_bd[:, 0], XYZ_left_bd[:, 1], XYZ_left_bd[:, 2])
            # uright = u_right(XYZ_right_bd[:, 0], XYZ_right_bd[:, 1], XYZ_right_bd[:, 2])
            # ufront = u_front(XYZ_front_bd[:, 0], XYZ_front_bd[:, 1], XYZ_front_bd[:, 2])
            # ubehind = u_behind(XYZ_behind_bd[:, 0], XYZ_behind_bd[:, 1], XYZ_behind_bd[:, 2])
            U_left = tf.constant(0.0)
            U_right = tf.constant(0.0)
            U_top = tf.constant(0.0)
            U_bottom = tf.constant(0.0)
            U_front = tf.constant(0.0)
            U_behind = tf.constant(0.0)

            loss_bd2UNN1 = tf.square(ULeft_NN1 - U_left) + tf.square(URight_NN1 - U_right) + \
                           tf.square(UBottom_NN1 - U_bottom) + tf.square(UTop_NN1 - U_top) + \
                           tf.square(UFront_NN1 - U_front) + tf.square(UBehind_NN1 - U_behind)
            loss_Ubd_NN1 = tf.reduce_mean(loss_bd2UNN1)

            loss_bd2UNN2 = tf.square(ULeft_NN2 - U_left) + tf.square(URight_NN2 - U_right) + \
                           tf.square(UBottom_NN2 - U_bottom) + tf.square(UTop_NN2 - U_top) + \
                           tf.square(UFront_NN2 - U_front) + tf.square(UBehind_NN2 - U_behind)
            loss_Ubd_NN2 = tf.reduce_mean(loss_bd2UNN2)

            # 边界上的导数
            # Psi_left = tf.reshape(u_left(XY_left_bd[:, 0], XY_left_bd[:, 1]), shape=[-1, 1])
            # Psi_right = tf.reshape(u_right(XY_right_bd[:, 0], XY_right_bd[:, 1]), shape=[-1, 1])
            # Psi_bottom = tf.reshape(u_bottom(XY_bottom_bd[:, 0], XY_bottom_bd[:, 1]), shape=[-1, 1])
            # Psi_top = tf.reshape(u_top(XY_top_bd[:, 0], XY_top_bd[:, 1]), shape=[-1, 1])
            Psi_left = tf.constant(0.0)
            Psi_right = tf.constant(0.0)
            Psi_bottom = tf.constant(0.0)
            Psi_top = tf.constant(0.0)
            Psi_front = tf.constant(0.0)
            Psi_behind = tf.constant(0.0)

            loss_bd_2dderivative_NN1 = tf.square(PsiLeft_NN1 - Psi_left) + tf.square(PsiRight_NN1 - Psi_right) + \
                                       tf.square(PsiBottom_NN1 - Psi_bottom) + tf.square(PsiTop_NN1 - Psi_top) + \
                                       tf.square(PsiFront_NN1 - Psi_front) + tf.square(PsiBehind_NN1 - Psi_behind)
            loss_Psibd_NN1 = tf.reduce_mean(loss_bd_2dderivative_NN1)

            loss_bd_2dderivative_NN2 = tf.square(PsiLeft_NN2 - Psi_left) + tf.square(PsiRight_NN2 - Psi_right) + \
                                       tf.square(PsiBottom_NN2 - Psi_bottom) + tf.square(PsiTop_NN2 - Psi_top) + \
                                       tf.square(PsiFront_NN2 - Psi_front) + tf.square(PsiBehind_NN2 - Psi_behind)
            loss_Psibd_NN2 = tf.reduce_mean(loss_bd_2dderivative_NN2)

            if R['regular_weight_model'] == 'L1':
                regular_WBPsi_NN1 = DNN_base.regular_weights_biases_L1(Weights2psi_NN1, Bias2psi_NN1)
                regular_WBU_NN1 = DNN_base.regular_weights_biases_L1(Weights2u_NN1, Bias2u_NN1)
                regular_WBPsi_NN2 = DNN_base.regular_weights_biases_L1(Weights2psi_NN2, Bias2psi_NN2)
                regular_WBU_NN2 = DNN_base.regular_weights_biases_L1(Weights2u_NN2, Bias2u_NN2)
            elif R['regular_weight_model'] == 'L2':
                regular_WBPsi_NN1 = DNN_base.regular_weights_biases_L2(Weights2psi_NN1, Bias2psi_NN1)    # 正则化权重参数 L2正则化
                regular_WBU_NN1 = DNN_base.regular_weights_biases_L2(Weights2u_NN1, Bias2u_NN1)
                regular_WBPsi_NN2 = DNN_base.regular_weights_biases_L2(Weights2psi_NN2, Bias2psi_NN2)
                regular_WBU_NN2 = DNN_base.regular_weights_biases_L2(Weights2u_NN2, Bias2u_NN2)
            else:
                regular_WBPsi_NN1 = tf.constant(0.0)
                regular_WBU_NN1 = tf.constant(0.0)
                regular_WBPsi_NN2 = tf.constant(0.0)
                regular_WBU_NN2 = tf.constant(0.0)

            PWB_NN1 = wb_penalty * (regular_WBPsi_NN1 + regular_WBU_NN1)
            PWB_NN2 = wb_penalty * (regular_WBPsi_NN2 + regular_WBU_NN2)
            loss2Psi_NN1 = loss_it2Psi_NN1 + bd_penalty * loss_Psibd_NN1
            loss2U_NN1 = loss_it2U_NN1 + bd_penalty * loss_Ubd_NN1
            loss_NN1 = loss2Psi_NN1 + loss2U_NN1 + PWB_NN1

            loss2Psi_NN2 = loss_it2Psi_NN2 + bd_penalty * loss_Psibd_NN2
            loss2U_NN2 = loss_it2U_NN2 + bd_penalty * loss_Ubd_NN2
            loss_NN2 = loss2Psi_NN2 + loss2U_NN2 + PWB_NN2

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_op1_NN1 = my_optimizer.minimize(loss2Psi_NN1, global_step=global_steps)
            train_op2_NN1 = my_optimizer.minimize(loss2U_NN1, global_step=global_steps)
            train_op3_NN1 = my_optimizer.minimize(loss_NN1, global_step=global_steps)
            train_loss_NN1 = tf.group(train_op1_NN1, train_op2_NN1, train_op3_NN1)

            train_op1_NN2 = my_optimizer.minimize(loss2Psi_NN2, global_step=global_steps)
            train_op2_NN2 = my_optimizer.minimize(loss2U_NN2, global_step=global_steps)
            train_op3_NN2 = my_optimizer.minimize(loss_NN2, global_step=global_steps)
            train_loss_NN2 = tf.group(train_op1_NN2, train_op2_NN2, train_op3_NN2)

            U_true = u_true(X_it, Y_it, Z_it)
            train_mse_NN1 = tf.reduce_mean(tf.square(U_true - U_NN1))
            train_rel_NN1 = train_mse_NN1 / tf.reduce_mean(tf.square(U_true))

            train_mse_NN2 = tf.reduce_mean(tf.square(U_true - U_NN2))
            train_rel_NN2 = train_mse_NN2 / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    lossU_NN1, lossPsi_NN1, lossUBD_NN1, lossPsiBD_NN1, loss_All2NN1, trainMse_NN1, trainRel_NN1 = [], [], [], [], [], [], []
    lossU_NN2, lossPsi_NN2, lossUBD_NN2, lossPsiBD_NN2, loss_All2NN2, trainMse_NN2, trainRel_NN2 = [], [], [], [], [], [], []
    test_mse_NN1, test_rel_NN1 = [], []
    test_mse_NN2, test_rel_NN2 = [], []
    test_epoch = []

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
    test_xyz_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
    saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xyz_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyz_bottom_batch, xyz_top_batch, xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch = \
                DNN_data.rand_bd_3D(batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
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

            _, lossU_nn1, lossPsi_nn1, loss_Ubd_nn1, loss_Psibd_nn1, loss_nn1, train_mse_nn1, train_rel_nn1 = sess.run(
                [train_loss_NN1, loss2U_NN1, loss2Psi_NN1, loss_Ubd_NN1, loss_Psibd_NN1, loss_NN1, train_mse_NN1,
                 train_rel_NN1], feed_dict={XYZ_it: xyz_it_batch, XYZ_bottom_bd: xyz_bottom_batch,
                                            XYZ_top_bd: xyz_top_batch, XYZ_left_bd: xyz_left_batch,
                                            XYZ_right_bd: xyz_right_batch, XYZ_front_bd: xyz_front_batch,
                                            XYZ_behind_bd: xyz_behind_batch, in_learning_rate: tmp_lr,
                                            train_opt: train_option, bd_penalty: temp_penalty_bd})

            lossU_NN1.append(lossU_nn1)
            lossPsi_NN1.append(lossPsi_nn1)
            lossUBD_NN1.append(loss_Ubd_nn1)
            lossPsiBD_NN1.append(loss_Psibd_nn1)
            loss_All2NN1.append(loss_nn1)
            trainMse_NN1.append(train_mse_nn1)
            trainRel_NN1.append(train_rel_nn1)

            _, lossU_nn2, lossPsi_nn2, lossUbd_nn2, lossPsibd_nn2, loss_nn2, train_mse_nn2, train_rel_nn2 = sess.run(
                [train_loss_NN2, loss2U_NN2, loss2Psi_NN2, loss_Ubd_NN2, loss_Psibd_NN2, loss_NN2, train_mse_NN2,
                 train_rel_NN2], feed_dict={XYZ_it: xyz_it_batch, XYZ_bottom_bd: xyz_bottom_batch,
                                            XYZ_top_bd: xyz_top_batch, XYZ_left_bd: xyz_left_batch,
                                            XYZ_right_bd: xyz_right_batch, XYZ_front_bd: xyz_front_batch,
                                            XYZ_behind_bd: xyz_behind_batch, in_learning_rate: tmp_lr,
                                            train_opt: train_option, bd_penalty: temp_penalty_bd})
            lossU_NN2.append(lossU_nn2)
            lossPsi_NN2.append(lossPsi_nn2)
            lossUBD_NN2.append(lossUbd_nn2)
            lossPsiBD_NN2.append(lossPsibd_nn2)
            loss_All2NN2.append(loss_nn2)
            trainMse_NN2.append(train_mse_nn2)
            trainRel_NN2.append(train_rel_nn2)

            if i_epoch % 1000 == 0:
                pwb = 0.0
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb, lossU_nn1, loss_Ubd_nn1,
                                    loss_Psibd_nn1, loss_nn1, train_mse_nn1, train_rel_nn1, log_out=log_fileout_NN1)

                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb, lossU_nn2, lossUbd_nn2,
                                    lossPsibd_nn2, loss_nn2, train_mse_nn2, train_rel_nn2, log_out=log_fileout_NN2)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_NN12test, u_NN22test = sess.run(
                    [U_true, U_NN1, U_NN2], feed_dict={XYZ_it: test_xyz_bach, train_opt: train_option})
                point_square_error_NN1 = np.square(u_true2test - u_NN12test)
                test_mse2nn1 = np.mean(point_square_error_NN1)
                test_mse_NN1.append(test_mse2nn1)
                test_rel2nn1 = test_mse2nn1 / np.mean(np.square(u_true2test))
                test_rel_NN1.append(test_rel2nn1)

                point_square_error_NN2 = np.square(u_true2test - u_NN22test)
                test_mse2nn2 = np.mean(point_square_error_NN2)
                test_mse_NN2.append(test_mse2nn2)
                test_rel2nn2 = test_mse2nn2 / np.mean(np.square(u_true2test))
                test_rel_NN2.append(test_rel2nn2)
                print('mean square error of predict and real for testing: %10f' % test_mse2nn1)
                print('residual error of predict and real for testing: %10f\n' % test_rel2nn1)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % test_mse2nn1,
                                     log_fileout_NN1)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % test_rel2nn1,
                                     log_fileout_NN1)

                print('mean square error of predict and real for testing: %10f' % test_mse2nn2)
                print('residual error of predict and real for testing: %10f\n' % test_rel2nn2)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % test_mse2nn2,
                                     log_fileout_NN2)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % test_rel2nn2,
                                     log_fileout_NN2)

        # -----------------------save the results into mat, then plot them ---------------------
        saveData.save_trainLoss2mat_1actFunc_Navier(
            lossU_NN1, lossUBD_NN1, lossPsi_NN1, lossPsiBD_NN1, loss_All2NN1, actName=act_func1,
            outPath=R['FolderName'])
        saveData.save_trainLoss2mat_1actFunc_Navier(
            lossU_NN2, lossUBD_NN2, lossPsi_NN2, lossPsiBD_NN2, loss_All2NN2, actName=act_func2,
            outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossU_NN1, lossU_NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_U', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossPsi_NN1, lossPsi_NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_Psi', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossUBD_NN1, lossUBD_NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_Ubd', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossPsiBD_NN1, lossPsiBD_NN2, lossName1=act_func1,
                                            lossName2=act_func2,
                                            lossType='loss_Psibd', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(loss_All2NN1, loss_All2NN2, lossName1=act_func1,
                                            lossName2=act_func2,
                                            lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(trainMse_NN1, trainRel_NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(trainMse_NN2, trainRel_NN2, actName=act_func2, outPath=R['FolderName'])
        plotData.plotTrain_MSEs_RELs_2act_funcs(trainMse_NN1, trainMse_NN2, trainRel_NN1, trainRel_NN2,
                                                actName1=act_func1, actName2=act_func2, seedNo=R['seed'],
                                                outPath=R['FolderName'], yaxis_scale=True)

        # ------------------------------ plot the testing result -------------------------------
        saveData.save_3testSolus2mat(u_true2test, u_NN12test, u_NN22test, actName='Utrue', actName1=act_func1,
                                     actName2=act_func2, outPath=R['FolderName'])
        if R['hot_power'] == 1:
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                            seedNo=R['seed'], outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(u_NN12test, size_vec2mat=size2test, actName=act_func1,
                                            seedNo=R['seed'], outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(u_NN22test, size_vec2mat=size2test, actName=act_func2,
                                            seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_NN1, test_rel_NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse_NN2, test_rel_NN2, actName=act_func2, outPath=R['FolderName'])
        plotData.plot_Test_MSE_REL_2ActFuncs(test_mse_NN1, test_mse_NN2, test_rel_NN1, test_rel_NN2,
                                             epoches=test_epoch,
                                             actName1=act_func1, actName2=act_func2, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_square_error_NN1, actName=act_func1,
                                              outPath=R['FolderName'])
        saveData.save_test_point_wise_err2mat(point_square_error_NN2, actName=act_func2,
                                              outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_error_NN1, size_vec2mat=size2test, actName=act_func1,
                                         seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_error_NN2, size_vec2mat=size2test, actName=act_func2,
                                         seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

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
    R['eqs_type'] = 'Biharmonic_3D'
    R['eqs_name'] = 'PDE1'

    R['input_dim'] = 3                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['variational_loss'] = 1             # PDE变分
    R['Navier_boundary'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['hot_power'] = 1
    # R['batch_size2interior'] = 7500       # 内部训练数据的批大小
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000        # 边界训练数据的批大小
    R['regular_weight_model'] = 'L2'

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

    # R['hidden_layers'] = (5, 4, 4, 3, 2, 2)
    # R['hidden_layers'] = (40, 40, 30, 20, 20, 10)    # 2D问题选这个时，s2relu的效果显示不出来
    # R['hidden_layers'] = (60, 40, 40, 30, 20, 20)    # 3D问题选这个时，s2relu的效果好于tanh
    R['hidden_layers'] = (80, 60, 60, 40, 30, 30)      # 3D问题选这个时，s2relu的效果不如tanh
    # R['hidden_layers'] = (100, 100, 80, 60, 60, 40)
    # R['hidden_layers'] = (150, 150, 100, 80, 80, 50)
    # R['hidden_layers'] = (200, 100, 100, 80, 50, 50)
    # R['hidden_layers'] = (300, 200, 200, 100, 80, 80, 50)
    # R['hidden_layers'] = (400, 300, 300, 200, 100, 100, 50)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100, 100)
    # R['hidden_layers'] = (600, 400, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)

    # 网络模型的选择
    R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scale'

    # 激活函数的选择
    # R['act_name2NN1'] = 'relu'
    # R['act_name2NN1'] = '2relu'
    R['act_name2NN1'] = 'tanh'
    # R['act_name2NN1'] = 'sin'
    # R['act_name2NN1'] = 's2relu'

    # R['act_name2NN2'] = 'relu'
    # R['act_name2NN2']' = leaky_relu'
    # R['act_name2NN2'] = 'srelu'
    R['act_name2NN2'] = 's2relu'
    # R['act_name2NN2'] = 'powsin_srelu'
    # R['act_name2NN2'] = 'slrelu'
    # R['act_name2NN2'] = 'elu'
    # R['act_name2NN2'] = 'selu'
    # R['act_name2NN2'] = 'phi'

    solve_Biharmonic3D(R)
