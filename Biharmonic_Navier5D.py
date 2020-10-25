"""
@author: LXA
Benchmark Code of Laplace equations.

"""
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
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


def solve_Biharmonic5D(R):
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
    if R['eqs_type'] == 'Biharmonic_5D':
        # laplace laplace u = f
        f, u_true = Biharmonic_eqs.get_biharmonic_Navier_5D(input_dim=input_dim, out_dim=out_dim, left_bottom=region_lb,
                                                            right_top=region_rt, laplace_name=R['eqs_name'])

    flag2u = 'WB2u'
    flag2psi = 'WB2psi'
    hidden_layers = R['hidden_layers']
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
            XYZST_it = tf.placeholder(tf.float32, name='XYZST', shape=[None, input_dim])
            XYZST00 = tf.placeholder(tf.float32, name='XYZST00', shape=[None, input_dim])
            XYZST01 = tf.placeholder(tf.float32, name='XYZST01', shape=[None, input_dim])
            XYZST10 = tf.placeholder(tf.float32, name='XYZST10', shape=[None, input_dim])
            XYZST11 = tf.placeholder(tf.float32, name='XYZST11', shape=[None, input_dim])
            XYZST20 = tf.placeholder(tf.float32, name='XYZST20', shape=[None, input_dim])
            XYZST21 = tf.placeholder(tf.float32, name='XYZST21', shape=[None, input_dim])
            XYZST30 = tf.placeholder(tf.float32, name='XYZST30', shape=[None, input_dim])
            XYZST31 = tf.placeholder(tf.float32, name='XYZST31', shape=[None, input_dim])
            XYZST40 = tf.placeholder(tf.float32, name='XYZST40', shape=[None, input_dim])
            XYZST41 = tf.placeholder(tf.float32, name='XYZST41', shape=[None, input_dim])
            bd_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            if R['model'] == 'PDE_DNN':
                Psi_NN = DNN_base.PDE_DNN(XYZST_it, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi00_NN = DNN_base.PDE_DNN(XYZST00, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi01_NN = DNN_base.PDE_DNN(XYZST01, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi10_NN = DNN_base.PDE_DNN(XYZST10, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi11_NN = DNN_base.PDE_DNN(XYZST11, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi20_NN = DNN_base.PDE_DNN(XYZST20, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi21_NN = DNN_base.PDE_DNN(XYZST21, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi30_NN = DNN_base.PDE_DNN(XYZST30, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi31_NN = DNN_base.PDE_DNN(XYZST31, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi40_NN = DNN_base.PDE_DNN(XYZST40, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)
                Psi41_NN = DNN_base.PDE_DNN(XYZST41, Weights2psi, Biases2psi, hidden_layers, activate_name=act_func)

                U_NN = DNN_base.PDE_DNN(XYZST_it, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U00_NN = DNN_base.PDE_DNN(XYZST00, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U01_NN = DNN_base.PDE_DNN(XYZST01, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U10_NN = DNN_base.PDE_DNN(XYZST10, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U11_NN = DNN_base.PDE_DNN(XYZST11, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U20_NN = DNN_base.PDE_DNN(XYZST20, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U21_NN = DNN_base.PDE_DNN(XYZST21, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U30_NN = DNN_base.PDE_DNN(XYZST30, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U31_NN = DNN_base.PDE_DNN(XYZST31, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U40_NN = DNN_base.PDE_DNN(XYZST40, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
                U41_NN = DNN_base.PDE_DNN(XYZST41, Weights2u, Biases2u, hidden_layers, activate_name=act_func)
            elif R['model'] == 'PDE_DNN_scale':
                freq = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)
                Psi_NN = DNN_base.PDE_DNN_scale(XYZST_it, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi00_NN = DNN_base.PDE_DNN_scale(XYZST00, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi01_NN = DNN_base.PDE_DNN_scale(XYZST01, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi10_NN = DNN_base.PDE_DNN_scale(XYZST10, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi11_NN = DNN_base.PDE_DNN_scale(XYZST11, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi20_NN = DNN_base.PDE_DNN_scale(XYZST20, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi21_NN = DNN_base.PDE_DNN_scale(XYZST21, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi30_NN = DNN_base.PDE_DNN_scale(XYZST30, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi31_NN = DNN_base.PDE_DNN_scale(XYZST31, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi40_NN = DNN_base.PDE_DNN_scale(XYZST40, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)
                Psi41_NN = DNN_base.PDE_DNN_scale(XYZST41, Weights2psi, Biases2psi, hidden_layers, freq, activate_name=act_func)

                U_NN = DNN_base.PDE_DNN_scale(XYZST_it, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U00_NN = DNN_base.PDE_DNN_scale(XYZST00, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U01_NN = DNN_base.PDE_DNN_scale(XYZST01, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U10_NN = DNN_base.PDE_DNN_scale(XYZST10, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U11_NN = DNN_base.PDE_DNN_scale(XYZST11, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U20_NN = DNN_base.PDE_DNN_scale(XYZST20, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U21_NN = DNN_base.PDE_DNN_scale(XYZST21, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U30_NN = DNN_base.PDE_DNN_scale(XYZST30, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U31_NN = DNN_base.PDE_DNN_scale(XYZST31, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U40_NN = DNN_base.PDE_DNN_scale(XYZST40, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)
                U41_NN = DNN_base.PDE_DNN_scale(XYZST41, Weights2u, Biases2u, hidden_layers, freq, activate_name=act_func)

            X_it = tf.reshape(XYZST_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZST_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZST_it[:, 2], shape=[-1, 1])
            S_it = tf.reshape(XYZST_it[:, 3], shape=[-1, 1])
            T_it = tf.reshape(XYZST_it[:, 4], shape=[-1, 1])

            dPsi_NN = tf.gradients(Psi_NN,  XYZST_it)[0]
            dU_NN = tf.gradients(U_NN,  XYZST_it)[0]
            if R['variational_loss'] != 0:
                laplace2psi = tf.reshape(tf.reduce_sum(tf.square(dPsi_NN), axis=-1), shape=[-1, 1])
                laplace2u = tf.reshape(tf.reduce_sum(tf.square(dU_NN), axis=-1), shape=[-1, 1])
                loss_it2psi = tf.reduce_mean(0.5*laplace2psi - tf.multiply(Psi_NN, f(X_it, Y_it, Z_it, S_it, T_it)))
                loss_it2u = tf.reduce_mean(0.5 * laplace2u - tf.multiply(Psi_NN, U_NN))

            U00 = tf.constant(0.0)
            U01 = tf.constant(0.0)
            U10 = tf.constant(0.0)
            U11 = tf.constant(0.0)
            U20 = tf.constant(0.0)
            U21 = tf.constant(0.0)
            U30 = tf.constant(0.0)
            U31 = tf.constant(0.0)
            U40 = tf.constant(0.0)
            U41 = tf.constant(0.0)

            loss_bd_square = tf.square(U00_NN - U00) + tf.square(U01_NN - U01) + tf.square(U10_NN - U10) + \
                             tf.square(U11_NN - U11) + tf.square(U20_NN - U20) + tf.square(U21_NN - U21) + \
                             tf.square(U30_NN - U30) + tf.square(U31_NN - U31) + tf.square(U40_NN - U40) + \
                             tf.square(U41_NN - U41)
            loss_bd2U_NN = tf.reduce_mean(loss_bd_square)

            # 边界上的偏导数
            Psi00 = tf.constant(0.0)
            Psi01 = tf.constant(0.0)
            Psi10 = tf.constant(0.0)
            Psi11 = tf.constant(0.0)
            Psi20 = tf.constant(0.0)
            Psi21 = tf.constant(0.0)
            Psi30 = tf.constant(0.0)
            Psi31 = tf.constant(0.0)
            Psi40 = tf.constant(0.0)
            Psi41 = tf.constant(0.0)
            loss_bd_square2second_derivative = tf.square(Psi00_NN - Psi00) + tf.square(Psi01_NN - Psi01) + \
                                               tf.square(Psi10_NN - Psi10) + tf.square(Psi11_NN - Psi11) + \
                                               tf.square(Psi20_NN - Psi20) + tf.square(Psi21_NN - Psi21) + \
                                               tf.square(Psi30_NN - Psi30) + tf.square(Psi31_NN - Psi31) + \
                                               tf.square(Psi40_NN - Psi40) + tf.square(Psi41_NN - Psi41)
            loss_bd2Psi_NN = tf.reduce_mean(loss_bd_square2second_derivative)

            if R['regular_weight_model'] == 'L1':
                regular_WB = DNN_base.regular_weights_biases_L1(Weights2u, Biases2u)    # 正则化权重参数 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB = DNN_base.regular_weights_biases_L2(Weights2u, Biases2u)    # 正则化权重参数 L2正则化
            else:
                regular_WB = 0.0                                                                   # 无正则化权重参数

            PWB = wb_penalty * regular_WB
            loss2Psi_NN = loss_it2psi + bd_penalty * loss_bd2Psi_NN
            loss2U_NN = loss_it2u + bd_penalty * loss_bd2U_NN
            loss_NN = loss2Psi_NN + loss2U_NN                                          # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_op1 = my_optimizer.minimize(loss2Psi_NN, global_step=global_steps)
            train_op2 = my_optimizer.minimize(loss2U_NN, global_step=global_steps)
            train_op3 = my_optimizer.minimize(loss_NN, global_step=global_steps)
            train_loss = tf.group(train_op1, train_op2, train_op3)

            U_true = u_true(X_it, Y_it, Z_it, S_it, T_it)
            train_mse = tf.reduce_mean(tf.square(U_true - U_NN))
            train_rel = train_mse / tf.reduce_mean(tf.square(U_true))

    t0 = time.time()
    lossU_all, lossPsi_all, loss_bd_all, loss_bdd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], [], [], []  # 空列表, 使用 append() 添加元素
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
    test_xyzst_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
    saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xyzst_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyzst00_batch, xyzst01_batch, xyzst10_batch, xyzst11_batch, xyzst20_batch, xyzst21_batch, xyzst30_batch, \
            xyzst31_batch, xyzst40_batch, xyzst41_batch = DNN_data.rand_bd_5D(batchsize_bd, input_dim,
                                                                              region_a=region_lb,
                                                                              region_b=region_rt)
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

            _, loss2u, loss2psi, loss2bd, loss2bdd, loss, train_mse_tmp, train_rel_tmp, pwb = sess.run(
                [train_loss, loss2U_NN, loss2Psi_NN, loss_bd2U_NN, loss_bd2Psi_NN, loss_NN, train_mse, train_rel, PWB]
                , feed_dict={XYZST_it: xyzst_it_batch, XYZST00: xyzst00_batch, XYZST01: xyzst01_batch,
                           XYZST10: xyzst10_batch, XYZST11: xyzst11_batch, XYZST20: xyzst20_batch,
                           XYZST21: xyzst21_batch, XYZST30: xyzst30_batch, XYZST31: xyzst31_batch,
                           XYZST40: xyzst40_batch, XYZST41: xyzst41_batch, bd_penalty: temp_penalty_bd,
                           train_opt: train_option})

            lossU_all.append(loss2u)
            lossPsi_all.append(loss2psi)
            loss_bd_all.append(loss2bd)
            loss_bdd_all.append(loss2bdd)
            loss_all.append(loss)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_bd, pwb, loss2u, loss2bd,
                                    loss2bdd, loss, train_mse_tmp, train_rel_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                u_true2test, u_nn2test = sess.run([U_true, U_NN], feed_dict={XYZST_it: test_xyzst_bach, train_opt: train_option})
                point_square_error = np.square(u_true2test - u_nn2test)
                mse2test = np.mean(point_square_error)
                test_mse_all.append(mse2test)
                rel2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(rel2test)
                print('mean square error of predict and real for testing: %10f' % mse2test)
                print('residual error of predict and real for testing: %10f\n' % rel2test)
                DNN_tools.log_string('mean square error of predict and real for testing: %10f' % mse2test, log_fileout)
                DNN_tools.log_string('residual error of predict and real for testing: %10f\n\n' % rel2test, log_fileout)

            if (i_epoch != 0) and (i_epoch != 100000) and (i_epoch % 10000 == 0):
                pathOut = '%s/%s' % (R['FolderName'], int(i_epoch / 10000))
                if not os.path.exists(pathOut):       # 判断路径是否已经存在
                    os.mkdir(pathOut)            # 无 log_out_path 路径，创建一个 log_out_path 路径
                # 绘解得热力图
                plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                                seedNo=R['seed'], outPath=pathOut)
                # 绘制预测解得热力图
                plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName=act_func,
                                                seedNo=R['seed'], outPath=pathOut)

                saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
                plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                          seedNo=R['seed'], outPath=pathOut, yaxis_scale=True)

                # 绘制误差的能量图
                saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=pathOut)
                plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
                                                 seedNo=R['seed'], outPath=pathOut)

        # -----------------------  save training result to mat file, then plot them ---------------------------------
        saveData.save_trainLoss2mat_1actFunc_Navier(lossU_all, loss_bd_all, lossPsi_all, loss_bdd_all, loss_all,
                                                    actName=act_func, outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(lossU_all, lossType='loss_u', seedNo=R['seed'],
                                          outPath=R['FolderName'])
        plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'],
                                          outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName='s2ReLU', outPath=R['FolderName'])
        plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName='s2ReLU', seedNo=R['seed'],
                                             outPath=R['FolderName'], yaxis_scale=True)

        # ------------------------------ save testing result to mat file, then plot them -------------------------------
        saveData.save_testData_or_solus2mat(u_true2test, dataName='Utrue', outPath=R['FolderName'])
        saveData.save_testData_or_solus2mat(u_nn2test, dataName=act_func, outPath=R['FolderName'])
        # 绘解得热力图
        plotData.plot_Hot_point_wise_err(u_true2test, size_vec2mat=size2test, actName='Utrue',
                                         seedNo=R['seed'], outPath=R['FolderName'])
        # 绘制预测解得热力图
        plotData.plot_Hot_point_wise_err(u_nn2test, size_vec2mat=size2test, actName='s2ReLU',
                                         seedNo=R['seed'], outPath=R['FolderName'])

        saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                                  seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

        # 绘制误差的能量图
        saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])
        plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
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
    R['eqs_type'] = 'Biharmonic_5D'
    # R['eqs_name'] = 'Navier_5D_1'
    R['eqs_name'] = 'Navier_5D_2'

    R['input_dim'] = 5                    # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                   # 输出维数
    R['variational_loss'] = 1             # PDE变分
    R['Navier_boundary'] = 1

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['hot_power'] = 1
    R['batch_size2interior'] = 10000       # 内部训练数据的批大小
    R['batch_size2boundary'] = 2000        # 边界训练数据的批大小
    # R['regular_weight_model'] = 'L2'
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
    elif 20000 < R['max_epoch'] and 50000 > R['max_epoch']:
        R['learning_rate'] = 1e-4  # 学习率
        R['lr_decay'] = 4e-5       # 学习率 decay
    else:
        R['learning_rate'] = 5e-5  # 学习率
        R['lr_decay'] = 1e-5       # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器

    # R['hidden_layers'] = (40, 40, 30, 20, 20, 10)
    # R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
    # R['hidden_layers'] = (100, 100, 80, 60, 60, 40)
    # R['hidden_layers'] = (200, 100, 100, 80, 50, 50)
    R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
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

    solve_Biharmonic5D(R)
