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
import DNN_boundary
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout, actName=None):
    DNN_tools.log_string('Equation type for problem: %s\n' % (R_dic['eqs_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(actName), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)

    if 1 == R['Dirichlet_boundary']:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Dirichlet boundary'), log_fileout)
    else:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Navier boundary'), log_fileout)
    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_bd_penalty']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)

    if R_dic['variational_loss'] == 1:
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: original function loss\n', log_fileout)


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


def solve_Biharmonic2D(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径

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
    if 'general_biharmonic' == str.lower(R['eqs_type']):
        # laplace laplace u = f
        f, u_true, u_left, u_right, u_bottom, u_top = Biharmonic_eqs.get_biharmonic_infos_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, laplace_name=R['eqs_name'])

    flag_NN1 = 'WB2NN1'
    flag_NN2 = 'WB2NN2'
    hidden_layers = R['hidden_layers']
    # Weights, Biases = Biharmonic_DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = Biharmonic_DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag)
    Weights2NN1, Bias2NN1 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag_NN1)
    Weights2NN2, Bias2NN2 = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag_NN2)

    act_func1 = R['act_name2NN1']
    act_func2 = R['act_name2NN2']
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_it = tf.placeholder(tf.float32, name='XY_it', shape=[None, input_dim])
            XY_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])
            XY_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])
            XY_bottom_bd = tf.placeholder(tf.float32, name='Y_bottom_bd', shape=[None, input_dim])
            XY_top_bd = tf.placeholder(tf.float32, name='Y_top_bd', shape=[None, input_dim])
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if 'PDE_DNN' == str.upper(R['model']):
                U_NN1 = DNN_base.PDE_DNN(XY_it, Weights2NN1, Bias2NN1, hidden_layers, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN(XY_left_bd, Weights2NN1, Bias2NN1, hidden_layers, activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN(XY_right_bd, Weights2NN1, Bias2NN1, hidden_layers, activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN(XY_bottom_bd, Weights2NN1, Bias2NN1, hidden_layers, activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN(XY_top_bd, Weights2NN1, Bias2NN1, hidden_layers, activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN(XY_it, Weights2NN2, Bias2NN2, hidden_layers, activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN(XY_left_bd, Weights2NN2, Bias2NN2, hidden_layers, activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN(XY_right_bd, Weights2NN2, Bias2NN2, hidden_layers, activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN(XY_bottom_bd, Weights2NN2, Bias2NN2, hidden_layers, activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN(XY_top_bd, Weights2NN2, Bias2NN2, hidden_layers, activate_name=act_func2)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                U_NN1 = DNN_base.PDE_DNN_scale(XY_it, Weights2NN1, Bias2NN1, hidden_layers, freq, activate_name=act_func1)
                ULeft_NN1 = DNN_base.PDE_DNN_scale(XY_left_bd, Weights2NN1, Bias2NN1, hidden_layers, freq,
                                                   activate_name=act_func1)
                URight_NN1 = DNN_base.PDE_DNN_scale(XY_right_bd, Weights2NN1, Bias2NN1, hidden_layers, freq,
                                                    activate_name=act_func1)
                UBottom_NN1 = DNN_base.PDE_DNN_scale(XY_bottom_bd, Weights2NN1, Bias2NN1, hidden_layers, freq,
                                                     activate_name=act_func1)
                UTop_NN1 = DNN_base.PDE_DNN_scale(XY_top_bd, Weights2NN1, Bias2NN1, hidden_layers, freq,
                                                  activate_name=act_func1)

                U_NN2 = DNN_base.PDE_DNN_scale(XY_it, Weights2NN2, Bias2NN2, hidden_layers, freq,
                                               activate_name=act_func2)
                ULeft_NN2 = DNN_base.PDE_DNN_scale(XY_left_bd, Weights2NN2, Bias2NN2, hidden_layers, freq,
                                                   activate_name=act_func2)
                URight_NN2 = DNN_base.PDE_DNN_scale(XY_right_bd, Weights2NN2, Bias2NN2, hidden_layers, freq,
                                                    activate_name=act_func2)
                UBottom_NN2 = DNN_base.PDE_DNN_scale(XY_bottom_bd, Weights2NN2, Bias2NN2, hidden_layers, freq,
                                                     activate_name=act_func2)
                UTop_NN2 = DNN_base.PDE_DNN_scale(XY_top_bd, Weights2NN2, Bias2NN2, hidden_layers, freq,
                                                  activate_name=act_func2)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])
            dU_NN1 = tf.gradients(U_NN1, XY_it)[0]
            dU_NN2 = tf.gradients(U_NN2, XY_it)[0]
            # 变分形式的loss of interior
            if 1 == R['variational_loss']:
                dU_NN1_x1 = tf.gather(dU_NN1, [0], axis=-1)
                dU_NN1_x2 = tf.gather(dU_NN1, [1], axis=-1)

                ddUNN1_x1 = tf.gradients(dU_NN1_x1, XY_it)[0]
                ddU_NN1_x1 = tf.gather(ddUNN1_x1, [0], axis=-1)

                ddUNN1_x2 = tf.gradients(dU_NN1_x2, XY_it)[0]
                ddU_NN1_x2 = tf.gather(ddUNN1_x2, [1], axis=-1)

                # ddU_NN1_x1x2 = tf.gather(ddU2ReLU_x1, [1], axis=-1)
                # ddU_hat_x1x2 = tf.gather(ddU_hat_x1, [1], axis=-1) + tf.gather(ddU_hat_x2, [0], axis=-1)

                laplace_U_NN1 = tf.square(ddU_NN1_x1) + tf.square(ddU_NN1_x2) + 2.0 * tf.multiply(ddU_NN1_x1, ddU_NN1_x2)

                temp=f(X_it, Y_it)
                loss_it2NN1 = 0.5 * laplace_U_NN1 - f(X_it, Y_it) * U_NN1
                loss_it_NN1 = tf.reduce_mean(loss_it2NN1)*np.power(region_rt - region_lb, input_dim)

                dU_NN2_x1 = tf.gather(dU_NN2, [0], axis=-1)
                dU_NN2_x2 = tf.gather(dU_NN2, [1], axis=-1)

                ddU2NN2_x1 = tf.gradients(dU_NN2_x1, XY_it)[0]
                ddU_NN2_x1 = tf.gather(ddU2NN2_x1, [0], axis=-1)

                ddU2NN2_x2 = tf.gradients(dU_NN2_x2, XY_it)[0]
                ddU_NN2_x2 = tf.gather(ddU2NN2_x2, [1], axis=-1)

                # ddU_NN2_x1x2 = tf.gather(ddU2NN2_x1, [1], axis=-1)
                # ddU_hat_x1x2 = tf.gather(ddU_hat_x1, [1], axis=-1) + tf.gather(ddU_hat_x2, [0], axis=-1)

                laplace_U_NN2 = tf.square(ddU_NN2_x1) + tf.square(ddU_NN2_x2) + 2.0 * tf.multiply(ddU_NN2_x1, ddU_NN2_x2)

                loss_it2NN2 = 0.5 * laplace_U_NN2 - f(X_it, Y_it) * U_NN2
                loss_it_NN2 = tf.reduce_mean(loss_it2NN2) * np.power(region_rt - region_lb, input_dim)

            # 边界loss，首先利用训练集把准确的边界值得到，然后和 neural  network 训练结果作差，最后平方
            # U_left = tf.reshape(u_left(XY_left_bd[:, 0], XY_left_bd[:, 1]), shape=[-1, 1])
            # U_right = tf.reshape(u_right(XY_right_bd[:, 0], XY_right_bd[:, 1]), shape=[-1, 1])
            # U_bottom = tf.reshape(u_bottom(XY_bottom_bd[:, 0], XY_bottom_bd[:, 1]), shape=[-1, 1])
            # U_top = tf.reshape(u_top(XY_top_bd[:, 0], XY_top_bd[:, 1]), shape=[-1, 1])
            U_left = tf.constant(0.0)
            U_right = tf.constant(0.0)
            U_top = tf.constant(0.0)
            U_bottom = tf.constant(0.0)
            loss_bd2NN1 = DNN_boundary.deal_0derivatives2NN_2d(ULeft_NN1, URight_NN1, UTop_NN1, UBottom_NN1, U_left,
                                                               U_right, U_top, U_bottom)
            loss_bd_NN1 = tf.reduce_mean(loss_bd2NN1)

            loss_bd2NN2 = DNN_boundary.deal_0derivatives2NN_2d(ULeft_NN2, URight_NN2, UTop_NN2, UBottom_NN2, U_left,
                                                               U_right, U_top, U_bottom)
            loss_bd_NN2 = tf.reduce_mean(loss_bd2NN2)

            # 边界上的偏导数(一阶偏导和二阶偏导)
            if 1 == R['Dirichlet_boundary']:
                U_left_1deriva = tf.constant(0.0)
                U_right_1deriva = tf.constant(0.0)
                U_top_1deriva = tf.constant(0.0)
                U_bottom_1deriva = tf.constant(0.0)

                loss2Dirichlet_NN1 = DNN_boundary.deal_1derivatives2NN_2d(
                    ULeft_NN1, URight_NN1, UTop_NN1, UBottom_NN1, U_left_1deriva, U_right_1deriva, U_top_1deriva,
                    U_bottom_1deriva, XY_left_bd, XY_right_bd, XY_top_bd, XY_bottom_bd)
                lossBD_derivative_NN1 = tf.reduce_mean(loss2Dirichlet_NN1)

                loss2Dirichlet_NN2 = DNN_boundary.deal_1derivatives2NN_2d(
                    ULeft_NN2, URight_NN2, UTop_NN2, UBottom_NN2, U_left_1deriva, U_right_1deriva, U_top_1deriva,
                    U_bottom_1deriva, XY_left_bd, XY_right_bd, XY_top_bd, XY_bottom_bd)
                lossBD_derivative_NN2 = tf.reduce_mean(loss2Dirichlet_NN2)

            if 1 == R['Navier_boundary']:
                U_left_2deriva = tf.constant(0.0)
                U_right_2deriva = tf.constant(0.0)
                U_top_2deriva = tf.constant(0.0)
                U_bottom_2deriva = tf.constant(0.0)

                loss2Navier_NN1 = DNN_boundary.deal_2derivatives2NN_2d(
                    ULeft_NN1, URight_NN1, UTop_NN1, UBottom_NN1, U_left_2deriva, U_right_2deriva, U_top_2deriva,
                    U_bottom_2deriva, XY_left_bd, XY_right_bd, XY_top_bd, XY_bottom_bd)
                lossBD_derivative_NN1 = tf.reduce_mean(loss2Navier_NN1)

                loss2Navier_NN2 = DNN_boundary.deal_2derivatives2NN_2d(
                    ULeft_NN2, URight_NN2, UTop_NN2, UBottom_NN2, U_left_2deriva, U_right_2deriva, U_top_2deriva,
                    U_bottom_2deriva, XY_left_bd, XY_right_bd, XY_top_bd, XY_bottom_bd)

                lossBD_derivative_NN2 = tf.reduce_mean(loss2Navier_NN2)

            if R['regular_weight_model'] == 'L1':
                regular_WB_NN1 = DNN_base.regular_weights_biases_L1(Weights2NN1, Bias2NN1)
                regular_WB_NN2 = DNN_base.regular_weights_biases_L1(Weights2NN2, Bias2NN2)
            elif R['regular_weight_model'] == 'L2':
                regular_WB_NN1 = DNN_base.regular_weights_biases_L2(Weights2NN1, Bias2NN1)
                regular_WB_NN2 = DNN_base.regular_weights_biases_L2(Weights2NN2, Bias2NN2)
            else:
                regular_WB_NN1 = tf.constant(0.0)
                regular_WB_NN2 = tf.constant(0.0)

            PWB_NN1 = wb_penalty * regular_WB_NN1
            PWB_NN2 = wb_penalty * regular_WB_NN2
            loss_NN1 = loss_it_NN1 + bd_penalty * (loss_bd_NN1 + lossBD_derivative_NN1) + PWB_NN1
            loss_NN2 = loss_it_NN2 + bd_penalty * (loss_bd_NN2 + lossBD_derivative_NN2) + PWB_NN2

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_loss_NN1 = my_optimizer.minimize(loss_NN1, global_step=global_steps)
            train_loss_NN2 = my_optimizer.minimize(loss_NN2, global_step=global_steps)

            U_true = u_true(X_it, Y_it)
            train_mse_NN1 = tf.reduce_mean(tf.square(U_true - U_NN1))
            train_rel_NN1 = train_mse_NN1/tf.reduce_mean(tf.square(U_true))

            train_mse_NN2 = tf.reduce_mean(tf.square(U_true - U_NN2))
            train_rel_NN2 = train_mse_NN2 / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    lossIt_all2NN1, lossBD_all2NN1, lossBD2_all2NN1, loss_all2NN1, train_mse_all2NN1, train_rel_all2NN1 = [], [], [], [], [], []
    test_mse_all2NN1, test_rel_all2NN1 = [], []
    lossIt_all2NN2, lossBD_all2NN2, lossBD2_all2NN2, loss_all2NN2, train_mse_all2NN2, train_res_all2NN2 = [], [], [], [], [], []
    test_mse_all2NN2, test_rel_all2NN2 = [], []
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
    test_xy_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = DNN_data.rand_bd_2D(
                batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if 1 == R['activate_stage_penalty']:
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
            elif 2 == R['activate_stage_penalty']:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < 2*int(R['max_epoch'] / 3):
                    temp_penalty_bd = 10 * bd_penalty_init
                else:
                    temp_penalty_bd = 50 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_nn1, loss_bd_nn1, loss_bd2_nn1, loss_nn1, train_mse_nn1, train_res_nn1, pwb_nn1 = sess.run(
                [train_loss_NN1, loss_it_NN1, loss_bd_NN1, lossBD_derivative_NN1, loss_NN1, train_mse_NN1,
                 train_rel_NN1, PWB_NN1], feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                 XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                 train_opt: train_option, bd_penalty: temp_penalty_bd})

            lossIt_all2NN1.append(loss_it_nn1)
            lossBD_all2NN1.append(loss_bd_nn1)
            lossBD2_all2NN1.append(loss_bd2_nn1)
            loss_all2NN1.append(loss_nn1)
            train_mse_all2NN1.append(train_mse_nn1)
            train_rel_all2NN1.append(train_res_nn1)

            _, loss_it_nn2, loss_bd_nn2, loss_bd2_nn2, loss_nn2, train_mse_nn2, train_res_nn2, pwb_nn2 = sess.run(
                [train_loss_NN2, loss_it_NN2, loss_bd_NN2, lossBD_derivative_NN2, loss_NN2, train_mse_NN2,
                 train_rel_NN2, PWB_NN2], feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                 XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                 train_opt: train_option, bd_penalty: temp_penalty_bd})

            lossIt_all2NN2.append(loss_it_nn2)
            lossBD_all2NN2.append(loss_bd_nn2)
            lossBD2_all2NN2.append(loss_bd2_nn2)
            loss_all2NN2.append(loss_nn2)
            train_mse_all2NN2.append(train_mse_nn2)
            train_res_all2NN2.append(train_res_nn2)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb_nn1, loss_it_nn1, loss_bd_nn1,
                                    loss_bd2_nn1, loss_nn1, train_mse_nn1, train_res_nn1, log_out=log_fileout_NN1)

                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb_nn2, loss_it_nn2, loss_bd_nn2,
                                    loss_bd2_nn2, loss_nn2, train_mse_nn2, train_res_nn2, log_out=log_fileout_NN2)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_NN1_test, u_NN2_test = sess.run(
                    [U_true, U_NN1, U_NN2], feed_dict={XY_it: test_xy_bach, train_opt: train_option})
                point_square_err2nn1 = np.square(u_true2test - u_NN1_test)
                test_mse_nn1 = np.mean(point_square_err2nn1)
                test_mse_all2NN1.append(test_mse_nn1)
                test_rel_nn1 = test_mse_nn1/np.mean(np.square(u_true2test))
                test_rel_all2NN1.append(test_rel_nn1)
                print('mean square error of predict and real for testing: %10f' % test_mse_nn1)
                print('residual error of predict and real for testing: %10f\n' % test_rel_nn1)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % test_mse_nn1, log_fileout_NN1)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % test_rel_nn1, log_fileout_NN1)

                point_square_err2nn2 = np.square(u_true2test - u_NN2_test)
                test_mse_nn2 = np.mean(point_square_err2nn2)
                test_mse_all2NN2.append(test_mse_nn2)
                test_rel_nn2 = test_mse_nn2 / np.mean(np.square(u_true2test))
                test_rel_all2NN2.append(test_rel_nn2)
                print('mean square error of predict and real for testing: %10f' % test_mse_nn2)
                print('residual error of predict and real for testing: %10f\n' % test_rel_nn2)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % test_mse_nn2,
                                     log_fileout_NN2)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % test_rel_nn2,
                                     log_fileout_NN2)

            if (i_epoch != 0) and (i_epoch != 100000) and (i_epoch % 10000 == 0) and (R['Navier_boundary'] == 1):
                pathOut = '%s/%s' % (R['FolderName'], int(i_epoch/10000))
                print('------- i_epoch-------:', i_epoch)
                print('\n')
                print('------- pathOut-------:', pathOut)
                if not os.path.exists(pathOut):       # 判断路径是否已经存在
                    os.mkdir(pathOut)            # 无 log_out_path 路径，创建一个 log_out_path 路径

                saveData.save_3testSolus2mat(u_true2test, u_NN1_test, u_NN2_test, actName1=act_func1, actName2=act_func2,
                                             outPath=pathOut)

                saveData.save_testMSE_REL2mat(test_mse_all2NN1, test_rel_all2NN1, actName=act_func1,
                                              outPath=pathOut)
                saveData.save_testMSE_REL2mat(test_mse_all2NN2, test_rel_all2NN2, actName=act_func2,
                                              outPath=pathOut)
                plotData.plot_Test_MSE_REL_2ActFuncs(test_mse_all2NN1, test_rel_all2NN1, test_mse_all2NN2,
                                                     test_rel_all2NN2, test_epoch, actName1=act_func1, actName2=act_func2,
                                                     seedNo=R['seed'], outPath=pathOut, yaxis_scale=True)

                saveData.save_test_point_wise_err2mat(point_square_err2nn1, actName=act_func1, outPath=pathOut)
                saveData.save_test_point_wise_err2mat(point_square_err2nn2, actName=act_func2, outPath=pathOut)

        # -----------------------  save training result to mat file, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc_Dirichlet(
            lossIt_all2NN1, lossBD_all2NN1, lossBD2_all2NN1, loss_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_trainLoss2mat_1actFunc_Dirichlet(
            lossIt_all2NN2, lossBD_all2NN2, lossBD2_all2NN2, loss_all2NN2, actName=act_func2, outPath=R['FolderName'])

        plotData.plotTrain_losses_2act_funs(lossIt_all2NN1, lossIt_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_losses_2act_funs(lossBD_all2NN1, lossBD_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTrain_losses_2act_funs(lossBD2_all2NN1, lossBD2_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss_bd2', seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTrain_losses_2act_funs(loss_all2NN1, loss_all2NN2, lossName1=act_func1, lossName2=act_func2,
                                            lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all2NN1, train_rel_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_train_MSE_REL2mat(train_mse_all2NN2, train_res_all2NN2, actName=act_func2, outPath=R['FolderName'])
        plotData.plotTrain_MSEs_2act_funcs(train_mse_all2NN1, train_mse_all2NN2, mseName1=act_func1, mseName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
        plotData.plotTrain_RELs_2act_funcs(train_mse_all2NN1, train_mse_all2NN2, relName1=act_func1, relName2=act_func2,
                                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # ------------------------------ save testing result to mat file, then plot them -------------------------------
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(u_NN2_test, dataName=act_func2, outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(u_NN1_test, dataName=act_func1, outPath=R['FolderName'])
        # saveData.save_3testSolus2mat(u_true2test, u_NN22test, U_NN12test, dataName='solution', outPath=R['FolderName'])

        # 绘解得热力图
        plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                        seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Hot_solution2test(u_NN1_test, size_vec2mat=size2test, actName=act_func1,
                                        seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Hot_solution2test(u_NN2_test, size_vec2mat=size2test, actName=act_func2,
                                        seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all2NN1, test_rel_all2NN1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse_all2NN2, test_rel_all2NN2, actName=act_func2, outPath=R['FolderName'])
        plotData.plot_Test_MSE_REL_2ActFuncs(test_mse_all2NN1, test_rel_all2NN1, test_mse_all2NN2,
                                             test_rel_all2NN2, test_epoch, actName1=act_func1, actName2=act_func2,
                                             seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # 绘制误差的能量图
        saveData.save_test_point_wise_err2mat(point_square_err2nn1, actName=act_func1, outPath=R['FolderName'])
        saveData.save_test_point_wise_err2mat(point_square_err2nn2, actName=act_func2, outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_err2nn1, size_vec2mat=size2test, actName=act_func1,
                                         seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_err2nn2, size_vec2mat=size2test, actName=act_func2,
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
    R['eqs_type'] = 'general_Biharmonic'
    # R['eqs_name'] = 'Dirichlet1'
    R['eqs_name'] = 'Dirichlet1_1'
    # R['eqs_name'] = 'Dirichlet2'
    # R['eqs_name'] = 'Dirichlet3'
    # R['eqs_name'] = 'Dirichlet3_1'
    # R['eqs_name'] = 'Navier4'
    # R['eqs_name'] = 'Navier5'
    # R['eqs_name'] = 'Navier6'

    R['input_dim'] = 2                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['variational_loss'] = 1             # PDE变分
    if R['eqs_name'] == 'Navier4' or R['eqs_name'] == 'Navier5'or R['eqs_name'] == 'Navier6':
        R['Dirichlet_boundary'] = 0
        R['Navier_boundary'] = 1
    else:
        R['Dirichlet_boundary'] = 1
        R['Navier_boundary'] = 0

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['hot_power'] = 1
    R['batch_size2interior'] = 3000       # 内部训练数据的批大小
    R['batch_size2boundary'] = 500        # 边界训练数据的批大小

    R['init_bd_penalty'] = 500            # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_bd_penalty'] = 5

    R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    # R['regular_weight_model'] = 'L2'
    if R['regular_weight_model'] == 'L0':
        R['regular_weight'] = 0.000
    else:
        R['regular_weight'] = 0.0005         # Regularization parameter for weights

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    elif 20000 < R['max_epoch'] and 50000 >= R['max_epoch']:
        R['learning_rate'] = 1e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    else:
        R['learning_rate'] = 5e-5  # 学习率
        R['lr_decay'] = 1e-5       # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器

    if R['eqs_name'] == 'Dirichlet1_1' or R['eqs_name'] == 'Dirichlet2' or R['eqs_name'] == 'Dirichlet3_1':
        # R['hidden_layers'] = (10, 10, 8, 6, 6, 3)       # it is used to debug our work
        R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
        # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
    elif R['eqs_name'] == 'Navier4'or R['eqs_name'] == 'Navier5' or R['eqs_name'] == 'Navier6':
        # R['hidden_layers'] = (20, 20, 15, 10, 10, 4)    # it is used to debug our work
        R['hidden_layers'] = (100, 100, 80, 60, 60, 40)
        # R['hidden_layers'] = (200, 100, 100, 80, 50, 50)
        # R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
        # R['hidden_layers'] = (400, 300, 300, 200, 100, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100, 100)
        # R['hidden_layers'] = (600, 400, 400, 300, 200, 200, 100)
        # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)

    # 网络模型的选择
    R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_scale'

    # 激活函数的选择
    # R['act_name2NN1'] = 'relu'
    R['act_name2NN1'] = 'srelu'
    # R['act_name2NN1'] = 'sin'
    # R['act_name2NN1'] = 's2relu'

    # R['act_name'] = 'relu'
    # R['act_name']' = leaky_relu'
    # R['act_name'] = 'srelu'
    R['act_name2NN2'] = 's2relu'
    # R['act_name2NN2'] = 'powsin_srelu'
    # R['act_name'] = 'slrelu'
    # R['act_name'] = 'elu'
    # R['act_name'] = 'selu'
    # R['act_name'] = 'phi'

    solve_Biharmonic2D(R)
