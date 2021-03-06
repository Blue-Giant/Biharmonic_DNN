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
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Equation type for problem: %s\n' % (R_dic['eqs_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('activate function: %s\n' % str(R_dic['act_name']), log_fileout)
    DNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string(
            'optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']),
                             log_fileout)
    else:
        DNN_tools.log_string(
            'no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)

    if 1 == R['Navier_boundary']:
        DNN_tools.log_string('Boundary types to derivative: %s\n' % str('Navier boundary'), log_fileout)
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


def solve_Biharmonic_Navier2D(R):
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
    if 'general_biharmonic' == str.lower(R['eqs_type']):
        # laplace laplace u = f
        f, u_true, u_left, u_right, u_bottom, u_top = Biharmonic_eqs.get_biharmonic_Navier_2D(
            input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb, right_top=region_rt, laplace_name=R['eqs_name'])

    flag2u = 'WB2u'
    flag2psi = 'WB2psi'
    hidden_layers = R['hidden_layers']
    # Weights, Biases = ThinPlate_DNN_base.initialize_NN_xavier(input_dim, out_dim, hidden_layers, flag)
    # Weights, Biases = ThinPlate_DNN_base.initialize_NN_random_normal(input_dim, out_dim, hidden_layers, flag)
    Weights2u, Biases2u = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2u)
    Weights2psi, Biases2psi = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2psi)

    # Laplace Laplace u = f
    # Navier case: u = Laplace u=0
    # Dirichlet case: u = Bu/Bn=0
    # Laplace u = -psi <--> -Laplace u = psi
    # 那么我们有   - Laplace psi=f
    #              - Laplace u = psi
    #              u = partial u/partial n=0
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY_it = tf.placeholder(tf.float32, name='XY_it', shape=[None, input_dim])
            XY_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])
            XY_right_bd = tf.placeholder(tf.float32, name='X_right_bd', shape=[None, input_dim])
            XY_bottom_bd = tf.placeholder(tf.float32, name='Y_bottom_bd', shape=[None, input_dim])
            XY_top_bd = tf.placeholder(tf.float32, name='Y_top_bd', shape=[None, input_dim])
            boundary_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if 'PDE_DNN' == str.upper(R['model']):
                Psi_NN = DNN_base.PDE_DNN(XY_it, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                PsiLeft_NN = DNN_base.PDE_DNN(XY_left_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                PsiRight_NN = DNN_base.PDE_DNN(XY_right_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                PsiBottom_NN = DNN_base.PDE_DNN(XY_bottom_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                PsiTop_NN = DNN_base.PDE_DNN(XY_top_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)

                U_NN = DNN_base.PDE_DNN(XY_it, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN(XY_left_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN(XY_right_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                UBottom_NN = DNN_base.PDE_DNN(XY_bottom_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                UTop_NN = DNN_base.PDE_DNN(XY_top_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_BN' == str.upper(R['model']):
                Psi_NN = DNN_base.PDE_DNN_BN(XY_it, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func, is_training=train_opt)
                PsiLeft_NN = DNN_base.PDE_DNN_BN(XY_left_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func, is_training=train_opt)
                PsiRight_NN = DNN_base.PDE_DNN_BN(XY_right_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func, is_training=train_opt)
                PsiBottom_NN = DNN_base.PDE_DNN_BN(XY_bottom_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func, is_training=train_opt)
                PsiTop_NN = DNN_base.PDE_DNN_BN(XY_top_bd, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func, is_training=train_opt)
                U_NN = DNN_base.PDE_DNN_BN(XY_it, Weights2u, Biases2u, hidden_layers, activate_name=act_func, is_training=train_opt)
                ULeft_NN = DNN_base.PDE_DNN_BN(XY_left_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func, is_training=train_opt)
                URight_NN = DNN_base.PDE_DNN_BN(XY_right_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func, is_training=train_opt)
                UBottom_NN = DNN_base.PDE_DNN_BN(XY_bottom_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func, is_training=train_opt)
                UTop_NN = DNN_base.PDE_DNN_BN(XY_top_bd, Weights2u, Biases2u, hidden_layers, activate_name=act_func, is_training=train_opt)
            elif 'PDE_DNN_SCALE' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                Psi_NN = DNN_base.PDE_DNN_scale(XY_it, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                PsiLeft_NN = DNN_base.PDE_DNN_scale(XY_left_bd, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                PsiRight_NN = DNN_base.PDE_DNN_scale(XY_right_bd, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                PsiBottom_NN = DNN_base.PDE_DNN_scale(XY_bottom_bd, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                PsiTop_NN = DNN_base.PDE_DNN_scale(XY_top_bd, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                U_NN = DNN_base.PDE_DNN_scale(XY_it, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                ULeft_NN = DNN_base.PDE_DNN_scale(XY_left_bd, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                URight_NN = DNN_base.PDE_DNN_scale(XY_right_bd, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                UBottom_NN = DNN_base.PDE_DNN_scale(XY_bottom_bd, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                UTop_NN = DNN_base.PDE_DNN_scale(XY_top_bd, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])
            dPsi_NN = tf.gradients(Psi_NN,  XY_it)[0]
            dU_NN = tf.gradients(U_NN,  XY_it)[0]
            if R['variational_loss'] != 0:
                laplace2psi = tf.reshape(tf.reduce_sum(tf.square(dPsi_NN), axis=-1), shape=[-1, 1])
                laplace2u = tf.reshape(tf.reduce_sum(tf.square(dU_NN), axis=-1), shape=[-1, 1])
                loss_it2psi = tf.reduce_mean(0.5*laplace2psi - tf.multiply(Psi_NN, f(X_it, Y_it)))
                loss_it2u = tf.reduce_mean(0.5 * laplace2u - tf.multiply(Psi_NN, U_NN))

            # 边界loss
            # U_left = tf.reshape(u_left(XY_left_bd[:, 0], XY_left_bd[:, 1]), shape=[-1, 1])
            # U_right = tf.reshape(u_right(XY_right_bd[:, 0], XY_right_bd[:, 1]), shape=[-1, 1])
            # U_bottom = tf.reshape(u_bottom(XY_bottom_bd[:, 0], XY_bottom_bd[:, 1]), shape=[-1, 1])
            # U_top = tf.reshape(u_top(XY_top_bd[:, 0], XY_top_bd[:, 1]), shape=[-1, 1])
            U_left = tf.constant(0.0)
            U_right = tf.constant(0.0)
            U_top = tf.constant(0.0)
            U_bottom = tf.constant(0.0)

            loss_bd_square = tf.square(ULeft_NN - U_left) + tf.square(URight_NN - U_right) + \
                             tf.square(UBottom_NN - U_bottom) + tf.square(UTop_NN - U_top)
            loss_bd = tf.reduce_mean(loss_bd_square)

            # 边界上的偏导数
            if 0 == R['Navier_boundary']:
                dULeft_NN = tf.gradients(ULeft_NN, XY_left_bd)[0]
                dU_left = tf.gather(dULeft_NN, [0], axis=-1)

                dURight_NN = tf.gradients(URight_NN, XY_right_bd)[0]
                dU_right = tf.gather(dURight_NN, [0], axis=-1)

                dUBottom_NN = tf.gradients(UBottom_NN, XY_bottom_bd)[0]
                dU_bottom = tf.gather(dUBottom_NN, [1], axis=-1)

                dUTop_NN = tf.gradients(UTop_NN, XY_top_bd)[0]
                dU_top = tf.gather(dUTop_NN, [1], axis=-1)

                loss_1derivative = tf.square(dU_left) + tf.square(dU_right) + tf.square(dU_bottom) + tf.square(dU_top)
                loss_bd2derivative = tf.reduce_mean(loss_1derivative)
            else:
                loss_bd_square2second_derivative = tf.square(PsiLeft_NN) + tf.square(PsiRight_NN) + \
                                                   tf.square(PsiBottom_NN) + tf.square(PsiTop_NN)
                loss_bd2derivative = tf.reduce_mean(loss_bd_square2second_derivative)

            if R['regular_weight_model'] == 'L1':
                regular_WB2U = DNN_base.regular_weights_biases_L1(Weights2u, Biases2u)    # 正则化权重参数 L1正则化
                regular_WB2Psi = DNN_base.regular_weights_biases_L1(Weights2psi, Biases2psi)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2U = DNN_base.regular_weights_biases_L2(Weights2u, Biases2u)    # 正则化权重参数 L2正则化
                regular_WB2Psi = DNN_base.regular_weights_biases_L2(Weights2psi, Biases2psi)
            else:
                regular_WB2U = tf.constant(0.0)
                regular_WB2Psi = tf.constant(0.0)      # 无正则化权重参数

            PWB = wb_penalty * (regular_WB2U + regular_WB2Psi)
            loss2psi = loss_it2psi + boundary_penalty * loss_bd2derivative
            loss2u = loss_it2u + boundary_penalty * loss_bd
            loss = loss2psi + loss2u     # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_op1 = my_optimizer.minimize(loss2psi, global_step=global_steps)
            train_op2 = my_optimizer.minimize(loss2u, global_step=global_steps)
            train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
            train_loss_optimizer = tf.group(train_op1, train_op2, train_op3)

            U_true = u_true(X_it, Y_it)
            train_mse = tf.reduce_mean(tf.square(U_true - U_NN))
            train_rel = train_mse/tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    lossUit_all, lossPsiit_all, loss_bd_all, loss_bdd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
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
    test_xy_bach = DNN_data.rand_it(test_bach_size, 2, region_lb, region_rt)
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

            _, lossit2u, lossit2psi, loss_bd_tmp, loss_bd2_tmp, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_loss_optimizer, loss_it2u, loss_it2psi, loss_bd, loss_bd2derivative, loss, train_mse, train_rel,
                 PWB], feed_dict={XY_it: xy_it_batch, XY_left_bd: xl_bd_batch, XY_right_bd: xr_bd_batch,
                                  XY_bottom_bd: yb_bd_batch, XY_top_bd: yt_bd_batch, in_learning_rate: tmp_lr,
                                  train_opt: train_option, boundary_penalty: temp_penalty_bd})

            lossUit_all.append(lossit2u)
            lossPsiit_all.append(lossit2psi)
            loss_bd_all.append(loss_bd_tmp)
            loss_bdd_all.append(loss_bd2_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb, lossit2u, loss_bd_tmp,
                                    loss_bd2_tmp, loss_tmp, train_mse_tmp, train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_nn2test = sess.run([U_true, U_NN], feed_dict={XY_it: test_xy_bach, train_opt: train_option})
                point_square_error = np.square(u_true2test - u_nn2test)
                test_mse = np.mean(point_square_error)
                test_mse_all.append(test_mse)
                test_rel = test_mse/np.mean(np.square(u_true2test))
                test_rel_all.append(test_rel)
                print('mean square error of predict and real for testing: %10f' % test_mse)
                print('residual error of predict and real for testing: %10f\n' % test_rel)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % test_mse, log_fileout)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % test_rel, log_fileout)

        saveData.save_trainLoss2mat_1actFunc_Navier(lossUit_all, loss_bd_all, lossPsiit_all, loss_bdd_all, loss_all,
                                                    actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(lossUit_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ------------------------ save test results into mat, then plot the testing result --------------------------
        saveData.save_2testSolus2mat(u_true2test, u_nn2test, actName='Utrue', actName1=act_func, outPath=R['FolderName'])

        if R['hot_power'] == 0:
            plotData.plot_2solutions2test(u_true2test, u_nn2test, coord_points2test=test_xy_bach,
                                          batch_size2test=test_bach_size, seedNo=R['seed'], outPath=R['FolderName'],
                                          subfig_type=0)
        elif R['hot_power'] == 1:
            plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                            outPath=R['FolderName'])
            plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName=act_func, seedNo=R['seed'],
                                            outPath=R['FolderName'])
        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func, seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)

        saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func, seedNo=R['seed'],
                                         outPath=R['FolderName'])


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
    R['eqs_name'] = 'PDE4'
    # R['eqs_name'] = 'PDE5'
    # R['eqs_name'] = 'PDE6'

    R['input_dim'] = 2                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['variational_loss'] = 2             # PDE变分
    R['Navier_boundary'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['hot_power'] = 1
    R['batch_size2interior'] = 3000       # 内部训练数据的批大小
    R['batch_size2boundary'] = 500        # 边界训练数据的批大小
    R['regular_weight_model'] = 'L0'

    R['init_bd_penalty'] = 500           # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1       # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_bd_penalty'] = 5

    R['regular_weight'] = 0.000           # Regularization parameter for weights
    # R['regular_weight'] = 0.001         # Regularization parameter for weights
    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    elif 20000 < R['max_epoch'] and 50000 >= R['max_epoch']:
        R['learning_rate'] = 1e-4  # 学习率
        R['lr_decay'] = 5e-5       # 学习率 decay
    else:
        R['learning_rate'] = 5e-5  # 学习率
        R['lr_decay'] = 2e-5       # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器

    R['hidden_layers'] = (40, 40, 30, 20, 20, 10)
    # R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
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
    # R['act_name'] = 'relu'
    # R['act_name']' = leaky_relu'
    R['act_name'] = 'tanh'
    # R['act_name'] = 'srelu'
    # R['act_name'] = 's2relu'
    # R['act_name'] = 'slrelu'
    # R['act_name'] = 'elu'
    # R['act_name'] = 'selu'
    # R['act_name'] = 'phi'

    solve_Biharmonic_Navier2D(R)
