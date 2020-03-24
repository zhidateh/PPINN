#!/usr/bin/python3

"""
@author: Zhi Da, Teh
"""

import utils as utils
import os
import ppinn_2d_api as nn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler


setting_name        = "PPINN23"
node_filter         = 1
case_size           = 10    
sampling_ratio      = 0.00  #1% x 16208 x 10 (1% of total dataset)
bc_ratio            = 0.5   #10% x 16208 x 10 (10% of BC)
batch_ratio         = 1     #100% x 1% x 16208 x 10 (100% of sample)
empty_batch         = 1000    
num_epoch           = 50001
num_sampling        = 1
learning_rate       = 1e-3
wp                  = 1
ws                  = 1
sampling_method     = "case sampling"  #random sampling, grid sampling, case sampling, single case sampling, region sampling
preprocess_method   = "dimensionless"  #dimensionless / minmax
load_chkpt_path     = 0#'/home/zhida/Documents/PPINN/model/%s/%s-0'%("PPINN23","PPINN23")  ##use 0 to disable loading
train_model         = True  ##use False for quick prediction, True for training

##-----------------------------------Loading----------------------------------------------------
model_path  = os.getcwd() + '/2d_inviscid_model/%s/'%setting_name
test_path   = os.getcwd() + '/test/'
train_path  = os.getcwd() + '/train/'


#data is matrix of m x n, where m is number of dataset, n is number of nodes
P_back,x,y,P,rho,u,v,Et= utils.loadData(train_path + 'Domain/',node_filter)
P_back_w,x_w,y_w,P_w,rho_w,u_w,v_w,Et_w= utils.loadData(train_path + 'Wall/',node_filter)
P_back_o,x_o,y_o,P_o,rho_o,u_o,v_o,Et_o= utils.loadData(train_path + 'Outlet/',node_filter)
P_back_i,x_i,y_i,P_i,rho_i,u_i,v_i,Et_i= utils.loadData(train_path + 'Inlet/',node_filter)
P_back_c,x_c,y_c,P_c,rho_c,u_c,v_c,Et_c= utils.loadData(train_path + 'Centerline/',node_filter)

P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,Et_test = utils.loadData(test_path,node_filter)

#initiaise PPINN class
model = nn.PPINN_2D(P_back,x,y,P,rho,u,v,Et)

##Boundary -----------------------------------------------------------------
model.P_back_w  = P_back_w
model.x_w       = x_w
model.y_w       = y_w
model.P_w       = P_w
model.rho_w     = rho_w
model.u_w       = u_w
model.v_w       = v_w
model.Et_w      = Et_w

model.P_back_o = P_back_o
model.x_o      = x_o
model.y_o      = y_o
model.P_o      = P_o
model.rho_o    = rho_o
model.u_o      = u_o
model.v_o      = v_o
model.Et_o     = Et_o

model.P_back_i  = P_back_i
model.x_i       = x_i
model.y_i       = y_i
model.P_i       = P_i
model.rho_i     = rho_i
model.u_i       = u_i
model.v_i       = v_i
model.Et_i      = Et_i

model.P_back_c  = P_back_c
model.x_c       = x_c
model.y_c       = y_c
model.P_c       = P_c
model.rho_c     = rho_c
model.u_c       = u_c
model.v_c       = v_c
model.Et_c      = Et_c

##Pass important variable into the PPINN class
model.ckpt_name         = 'tmp/' + setting_name 
model.sampling_ratio    = sampling_ratio
model.bc_ratio          = bc_ratio
model.batch_ratio       = batch_ratio
model.case_size         = case_size
model.empty_batch       = empty_batch
model.num_epoch         = num_epoch
model.num_sampling      = num_sampling
model.learning_rate     = learning_rate
model.wp                = wp
model.ws                = ws
model.sampling_method   = sampling_method
model.preprocess_method = preprocess_method

#Functions
model.constructGraph()

#Determine whether to load model.  
if load_chkpt_path == 0:
    model.initGraph() 
else:
    saver = tf.compat.v1.train.import_meta_graph("%s.meta"%load_chkpt_path)
    saver.restore(model.sess,("%s"%load_chkpt_path))

if train_model:
    model.train()

##Testing and prediction ---------------------------------------------------------------------------------------------------------------
if P_back_test.shape[0] == 1:
    P_b_test_value  = P_back_test.flatten()[0]

    P_back_test = P_back_test.flatten()[:,None]
    x_test      = x_test.flatten()[:,None]
    y_test      = y_test.flatten()[:,None]
    P_test      = P_test.flatten()[:,None] 
    rho_test    = rho_test.flatten()[:,None]
    u_test      = u_test.flatten()[:,None]
    v_test      = v_test.flatten()[:,None]
    Et_test     = Et_test.flatten()[:,None]

    P_pred, rho_pred, u_pred, v_pred, Et_pred = model.predict(P_back_test,x_test,y_test)

    #prediction error
    error_P     = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
    error_rho   = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
    error_u     = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    error_v     = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
    error_Et    = np.linalg.norm(Et_test-Et_pred,2)/np.linalg.norm(Et_test,2)

    print("Average Test Error: %.3f"%((error_P+error_rho+error_u+error_v+error_Et)/5))
    print("P\trho\tu\tv\tEt")
    print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %(error_P,error_rho,error_u,error_v,error_Et))
    print("******************************************************")

    ##Save ---------------------------------------------------------------------------------------------------------------
    path = os.getcwd() + '/predict/%s_bp=%s.csv'%(setting_name,str(int(P_b_test_value)))
    utils.writeData(path,x_test,y_test,P_pred,rho_pred,u_pred,v_pred,Et_pred)

    path2 = os.getcwd() + '/predict/%s_bp=%s_gt_loss.csv'%(setting_name,str(int(P_b_test_value)))
    utils.writeLoss(path2,model.sse_loss_vector,model.step_vector)

    path3 = os.getcwd() + '/predict/%s_bp=%s_pinn_loss.csv'%(setting_name,str(int(P_b_test_value)))
    utils.writeLoss(path3,model.pinn_loss_vector,model.step_vector)


else:

    for case in range(P_back_test.shape[0]):
        P_b_test_value  = P_back_test[case:(case+1),:][0]

        P_back_test_ = P_back_test[case:(case+1),:].transpose()
        x_test_      = x_test[case:(case+1),:].transpose()
        y_test_      = y_test[case:(case+1),:].transpose()
        P_test_      = P_test[case:(case+1),:].transpose()
        rho_test_    = rho_test[case:(case+1),:].transpose()
        u_test_      = u_test[case:(case+1),:].transpose()
        v_test_      = v_test[case:(case+1),:].transpose()
        Et_test_     = Et_test[case:(case+1),:].transpose()

        P_pred, rho_pred, u_pred, v_pred, Et_pred = model.predict(P_back_test_,x_test_,y_test_)

        plt.plot(x_test_, P_test_, '-r')
        plt.plot(x_test_, P_pred, 'xb')
        plt.legend(['True value','Predicted value'],loc="best")

    plt.title('Pressure at Centerline')

    plt.show()