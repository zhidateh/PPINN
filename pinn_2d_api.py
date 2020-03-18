import tensorflow as tf
import numpy as np
import time
import math
import utils as utils 
from tensorflow.contrib.layers import fully_connected
from sklearn.preprocessing import MinMaxScaler

class PINN_2D:
    # Initialize the class
    def __init__(self, P_back, x, y, P, rho, u, v, Et):
                    
        self.sse_loss_vector, self.step_vector,self.pinn_loss_vector = [], [], []
       
        self.P_back     = P_back
        self.x          = x
        self.y          = y
        self.P          = P
        self.rho        = rho
        self.u          = u
        self.v          = v
        self.Et         = Et

    def preprocessData(self):

        if self.preprocess_method == 'dimensionless':
            cLength = 1
            cVelocity = 600

            #normalizer  
            self.rho_norm    = np.amax(self.rho)  
            self.x_norm      = cLength
            self.y_norm      = cLength
            self.P_norm      = self.rho_norm *cVelocity**2
            self.u_norm      = cVelocity
            self.v_norm      = cVelocity
            self.Et_norm     = cVelocity**2

            self.x          /= self.x_norm
            self.y          /= self.y_norm
            self.P_back     /= self.P_norm
            self.P          /= self.P_norm
            self.rho        /= self.rho_norm
            self.u          /= self.u_norm
            self.v          /= self.v_norm
            self.Et         /= self.Et_norm

            self.x_w       /= self.x_norm
            self.y_w       /= self.y_norm
            self.P_back_w  /= self.P_norm
            self.P_w       /= self.P_norm
            self.rho_w     /= self.rho_norm
            self.u_w       /= self.u_norm
            self.v_w       /= self.v_norm
            self.Et_w      /= self.Et_norm

            self.x_o       /= self.x_norm
            self.y_o       /= self.y_norm
            self.P_back_o  /= self.P_norm
            self.P_o       /= self.P_norm
            self.rho_o     /= self.rho_norm
            self.u_o       /= self.u_norm
            self.v_o       /= self.v_norm
            self.Et_o      /= self.Et_norm

            self.x_i       /= self.x_norm
            self.y_i       /= self.y_norm
            self.P_back_i  /= self.P_norm
            self.P_i       /= self.P_norm
            self.rho_i     /= self.rho_norm
            self.u_i       /= self.u_norm
            self.v_i       /= self.v_norm
            self.Et_i      /= self.Et_norm

            self.x_c       /= self.x_norm
            self.y_c       /= self.y_norm
            self.P_back_c  /= self.P_norm
            self.P_c       /= self.P_norm
            self.rho_c     /= self.rho_norm
            self.u_c       /= self.u_norm
            self.v_c       /= self.v_norm
            self.Et_c      /= self.Et_norm

            # print(self.P.max(),self.P.min())
            # print(self.rho.max(),self.rho.min())
            # print(self.u.max(),self.u.min())
            # print(self.v.max(),self.v.min())
            # print(self.Et.max(),self.Et.min())


        elif self.preprocess_method == 'minmax':
            self.p_model    = MinMaxScaler()
            self.rho_model  = MinMaxScaler()
            self.u_model    = MinMaxScaler()
            self.v_model    = MinMaxScaler()
            self.Et_model   = MinMaxScaler()

            self.P      = self.p_model.fit_transform(self.P)
            self.P_back = self.p_model.transform(self.P_back)
            self.rho    = self.rho_model.fit_transform(self.rho)
            self.u      = self.u_model.fit_transform(self.u)
            self.v      = self.v_model.fit_transform(self.v)
            self.Et     = self.Et_model.fit_transform(self.Et)

    def constructGraph(self):        
        self.preprocessData()

        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                        log_device_placement=True))
        
        #For PINN calculation
        self.P_back_PINN_tf    = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, self.case_size])
        self.x_PINN_tf         = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, self.case_size])
        self.y_PINN_tf         = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, self.case_size])

        #For SSE Calculation
        self.P_back_SSE_tf      = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.x_SSE_tf           = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.y_SSE_tf           = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.P_SSE_tf           = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.rho_SSE_tf         = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.u_SSE_tf           = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.v_SSE_tf           = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        self.Et_SSE_tf          = tf.compat.v1.placeholder(dtype =tf.float32, shape=[None, 1])
        
        ##Enable if want to try dropout
        # self.normalizer_fn = tf.contrib.layers.dropout
        # self.normalizer_params = {'keep_prob': 0.75}
        # self.hidden_layer_2    = fully_connected(self.hidden_layer_1,layers[1],normalizer_fn = self.normalizer_fn,normalizer_params= self.normalizer_params,activation_fn= self.actf,scope = 'l2',reuse=False)
              
        #Construct tf graph
        self.actf  = tf.tanh

        #Parallel PINN error
        self.e_1, self.e_2, self.e_3, self.e_4, self.e_5  = 0,0,0,0,0
        for case in range(self.case_size):
            #input matrix shape: mx1
            self.x_input            =  self.x_PINN_tf[:,case:case+1]         #slice from each case
            self.y_input            =  self.y_PINN_tf[:,case:case+1]         #slice from each case
            self.P_back_input       =  self.P_back_PINN_tf[:,case:case+1]    #slice from each case    

            self.input_layer        = tf.concat([self.P_back_input, self.x_input, self.y_input], 1)             
            self.hidden_layer_1     = fully_connected(self.input_layer   ,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l1',reuse=bool(case))
            self.hidden_layer_2     = fully_connected(self.hidden_layer_1,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l2',reuse=bool(case))
            self.hidden_layer_3     = fully_connected(self.hidden_layer_2,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l3',reuse=bool(case))
            self.output_pred_1      = fully_connected(self.hidden_layer_3,5,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l6',reuse=bool(case))

            #PDE loss and SSE loss
            e_1, e_2, e_3, e_4, e_5 = self.PINNFunction(case)

            #Accumulate error across all cases
            self.e_1    += e_1
            self.e_2    += e_2
            self.e_3    += e_3
            self.e_4    += e_4
            self.e_5    += e_5

        #SSE error
        self.input_layer        = tf.concat([self.P_back_SSE_tf, self.x_SSE_tf, self.y_SSE_tf], 1)             
        self.hidden_layer_1     = fully_connected(self.input_layer   ,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l1',reuse=True)
        self.hidden_layer_2     = fully_connected(self.hidden_layer_1,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l2',reuse=True)
        self.hidden_layer_3     = fully_connected(self.hidden_layer_2,15,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l3',reuse=True)
        self.output_pred_2      = fully_connected(self.hidden_layer_3,5,activation_fn= self.actf,weights_regularizer= tf.contrib.layers.l2_regularizer(0.005),scope = 'l6',reuse=True)
        self.e_P                = tf.reduce_sum(tf.square(self.output_pred_2[:,0:1] - self.P_SSE_tf))
        self.e_rho              = tf.reduce_sum(tf.square(self.output_pred_2[:,1:2] - self.rho_SSE_tf))
        self.e_u                = tf.reduce_sum(tf.square(self.output_pred_2[:,2:3] - self.u_SSE_tf))
        self.e_v                = tf.reduce_sum(tf.square(self.output_pred_2[:,3:4] - self.v_SSE_tf))
        self.e_Et               = tf.reduce_sum(tf.square(self.output_pred_2[:,4:5] - self.Et_SSE_tf))


        self.sse_loss    =  1*self.e_P      +\
                            1*self.e_rho  +\
                            1*self.e_u    +\
                            1*self.e_v    +\
                            1*self.e_Et 

        self.pinn_loss  =   1*self.e_1    +\
                            1*self.e_2    +\
                            1*self.e_3    +\
                            1*self.e_4    +\
                            1*self.e_5  


        self.loss   =   self.wp*self.pinn_loss +\
                        self.ws*self.sse_loss +\
                        0*tf.compat.v1.losses.get_regularization_loss() #Regularization to prevent overfitting

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)   

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 20000,
                                                                           'maxfun': 15000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
       
        self.saver = tf.train.Saver(save_relative_paths=True)

    def PINNFunction(self,case):
        gamma = 1.4
        e_1, e_2, e_3, e_4, e_5 = 0,0,0,0,0

        x_pred   = self.x_input
        y_pred   = self.y_input

        P_pred   = self.output_pred_1[:,0:1] 
        rho_pred = self.output_pred_1[:,1:2] 
        u_pred   = self.output_pred_1[:,2:3] 
        v_pred   = self.output_pred_1[:,3:4] 
        Et_pred  = self.output_pred_1[:,4:5] 

        #PINN pred error 
        residual_1      = tf.gradients(rho_pred   *u_pred, x_pred,unconnected_gradients='zero')[0] \
                        + tf.gradients(rho_pred   *v_pred, y_pred,unconnected_gradients='zero')[0] \
                        + rho_pred*v_pred/ y_pred
        
        residual_2      = tf.gradients(y_pred     *rho_pred   *u_pred *u_pred, x_pred,unconnected_gradients='zero')[0]/y_pred     \
                        + tf.gradients(y_pred     *rho_pred   *v_pred *u_pred, y_pred,unconnected_gradients='zero')[0]/y_pred   \
                        + tf.gradients(P_pred     ,x_pred     ,unconnected_gradients='zero')[0]
        
        residual_3      = tf.gradients(y_pred     *rho_pred   *v_pred *u_pred , x_pred,unconnected_gradients='zero')[0]/y_pred     \
                        + tf.gradients(y_pred     *rho_pred   *v_pred *v_pred , y_pred,unconnected_gradients='zero')[0]/y_pred  \
                        + tf.gradients(P_pred     ,y_pred     ,unconnected_gradients='zero')[0]
                    
        residual_4      = tf.gradients(rho_pred   *Et_pred    *u_pred , x_pred,unconnected_gradients='zero')[0]  \
                        + tf.gradients(rho_pred   *Et_pred    *v_pred , y_pred,unconnected_gradients='zero')[0] \
                        + tf.gradients(P_pred     *u_pred     , x_pred,unconnected_gradients='zero')[0] \
                        + tf.gradients(P_pred     *v_pred     , y_pred,unconnected_gradients='zero')[0] \
                        + (rho_pred * Et_pred + P_pred)*v_pred/y_pred
        
        residual_5       =  P_pred - (1 -1/gamma)*rho_pred*(Et_pred - 0.5*u_pred**2 - 0.5*v_pred**2) 
        

        e_1     = tf.reduce_mean(tf.square(residual_1))
        e_2     = tf.reduce_mean(tf.square(residual_2))
        e_3     = tf.reduce_mean(tf.square(residual_3))
        e_4     = tf.reduce_mean(tf.square(residual_4))
        e_5     = tf.reduce_mean(tf.square(residual_5))

        return e_1, e_2, e_3, e_4, e_5

    def callback(self, loss):
        # self.loss_vector.append(loss)
        # self.step_vector.append(1)
        print('Loss: %.3e' % (loss))

    def initGraph(self):
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def train(self):        
        start_time = time.time()

        for nSampling in range(self.num_sampling):
            P_back_SSE_sample, x_SSE_sample, y_SSE_sample, P_SSE_sample, rho_SSE_sample, u_SSE_sample, v_SSE_sample, Et_SSE_sample,\
            P_back_PINN_sample, x_PINN_sample, y_PINN_sample = self.generateSample()
            
            for nEpoch in range(self.num_epoch):

                for nIt in range( int(1.0/self.batch_ratio)):
                    batch_size = int(x_SSE_sample.shape[0] * self.batch_ratio)

                    tf_dict = { self.P_back_SSE_tf  : P_back_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.x_SSE_tf       : x_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.y_SSE_tf       : y_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.P_SSE_tf       : P_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.rho_SSE_tf     : rho_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.u_SSE_tf       : u_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.v_SSE_tf       : v_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.Et_SSE_tf      : Et_SSE_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.P_back_PINN_tf : P_back_PINN_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.x_PINN_tf      : x_PINN_sample[nIt*batch_size: (nIt+1)*batch_size,:],
                                self.y_PINN_tf      : y_PINN_sample[nIt*batch_size: (nIt+1)*batch_size,:]}

                    self.sess.run(self.train_op_Adam, tf_dict)

                    # Print
                    if nEpoch % 100 == 0:

                        elapsed = (time.time() - start_time)/60.0

                        loss_value                  = self.sess.run([self.loss],tf_dict)[0]
                        sse_loss_value              = self.sess.run([self.sse_loss],tf_dict)[0]
                        pinn_loss_value             = self.sess.run([self.pinn_loss],tf_dict)[0]
                        e_1, e_2, e_3,e_4,e_5       = self.sess.run([self.e_1,self.e_2,self.e_3,self.e_4, self.e_5],tf_dict)
                        e_P, e_rho, e_u,e_v,e_Et    = self.sess.run([self.e_P,self.e_rho,self.e_u,self.e_v,self.e_Et],tf_dict)

                        print("Sampling: %d, Epoch: %d, Elapsed: %.2f min" %(nSampling, nEpoch,elapsed))
                        print("Batch size: %d, PINN Batch size: %d" %(x_SSE_sample.shape[0] * self.batch_ratio,x_PINN_sample.flatten().shape[0]*self.batch_ratio))
                        print("-----PINN-----\t-----ANN-----")
                        print("E_1: %.3f \tE_P: %.3f "%(e_1,e_P))
                        print("E_2: %.3f \tE_r: %.3f "%(e_2,e_rho))
                        print("E_3: %.3f \tE_u: %.3f "%(e_3,e_u))
                        print("E_4: %.3f \tE_v: %.3f "%(e_4,e_v))
                        print("E_5: %.3f \tE_E: %.3f "%(e_5,e_Et))
                        print("Ground truth Error: %.3f" %(sse_loss_value))     
                        print("PINN Error        : %.3f\n" %(pinn_loss_value))     

                        self.saver.save(self.sess,self.ckpt_name,global_step = nSampling)
                        self.pinn_loss_vector.append(pinn_loss_value)
                        self.sse_loss_vector.append(sse_loss_value)
                        self.step_vector.append(1)

        # self.optimizer.minimize(self.sess,
        #                     feed_dict = tf_dict,
        #                     fetches = [self.loss],
        #                     loss_callback = self.callback)
       
    def predict(self, P_back_test, x_test, y_test):
        
        if self.preprocess_method == 'dimensionless':
            x_test       /= self.x_norm
            y_test       /= self.y_norm
            P_back_test  /= self.P_norm

        elif self.preprocess_method == 'minmax':
            P_back_test = self.p_model.transform(P_back_test.transpose()).transpose()

        tf_dict     = {self.P_back_SSE_tf: P_back_test, self.x_SSE_tf: x_test, self.y_SSE_tf: y_test}
        output_test = self.sess.run(self.output_pred_2,tf_dict)
        
        P_pred      = output_test[:,0:1]
        rho_pred    = output_test[:,1:2]
        u_pred      = output_test[:,2:3]
        v_pred      = output_test[:,3:4]
        Et_pred     = output_test[:,4:5]

        
        if self.preprocess_method == 'dimensionless':
            P_pred      *= self.P_norm
            rho_pred    *= self.rho_norm
            u_pred      *= self.u_norm
            v_pred      *= self.v_norm
            Et_pred     *= self.Et_norm

        elif self.preprocess_method == 'minmax':
            P_pred      = self.p_model.inverse_transform(P_pred.transpose()).transpose()                                       
            rho_pred    = self.rho_model.inverse_transform(rho_pred.transpose()).transpose()
            u_pred      = self.u_model.inverse_transform(u_pred.transpose()).transpose()
            v_pred      = self.v_model.inverse_transform(v_pred.transpose()).transpose()
            Et_pred     = self.Et_model.inverse_transform(Et_pred.transpose()).transpose()

        return P_pred, rho_pred, u_pred, v_pred, Et_pred

    def generateSample(self):
        selected_nodes = np.array([],dtype = int)

        if self.sampling_method == "case sampling":
            #initiate empty array
            P_back_SSE_batch   = np.array([])
            x_SSE_batch        = np.array([])
            y_SSE_batch        = np.array([])
            P_SSE_batch        = np.array([])
            rho_SSE_batch      = np.array([])
            u_SSE_batch        = np.array([])
            v_SSE_batch        = np.array([])
            Et_SSE_batch       = np.array([])
            P_back_PINN_batch  = np.array([])
            x_PINN_batch       = np.array([])
            y_PINN_batch       = np.array([])
            P_PINN_batch       = np.array([])
            rho_PINN_batch     = np.array([])
            u_PINN_batch       = np.array([])
            v_PINN_batch       = np.array([])
            Et_PINN_batch      = np.array([])

            if self.case_size  <= self.P.shape[0]:
                cases = np.random.choice(range(self.P.shape[0]), self.case_size , replace=False)
            else:
                cases = np.random.choice(range(self.P.shape[0]), self.case_size , replace=True)
            
            for case in range(len(cases)):
                i = cases[case]

                batch_per_case          = int(self.sampling_ratio * self.x.shape[1])
                batch_empty_per_case    = int(self.empty_batch/self.case_size)

                batch_logic             = np.logical_and( self.x[i] >= 0, self.x[i] <= 0.84 ) #converging region of domain
                empty_batch_logic       = np.logical_and( self.x[i] >= 0, self.x[i] <= 0.84 ) #entire domain

                true_batch_per_case  = batch_per_case + batch_empty_per_case

                #check if the logic 1 result in empty array
                if (self.P[i][batch_logic].shape[0] != 0):
                    if batch_per_case  <= self.P[i][batch_logic].shape[0]:
                        A = np.random.choice(self.P[i][batch_logic].shape[0], size=batch_per_case , replace=False)
                    else:
                        A = np.random.choice(self.P[i][batch_logic].shape[0], size=batch_per_case , replace=True)
                    
                    if(len(A)>0):
                        P_back_SSE_batch    = np.hstack((P_back_SSE_batch, self.P_back[i][batch_logic][A]))
                        x_SSE_batch         = np.hstack((x_SSE_batch, self.x[i][batch_logic][A])) 
                        y_SSE_batch         = np.hstack((y_SSE_batch, self.y[i][batch_logic][A]))
                        P_SSE_batch         = np.hstack((P_SSE_batch, self.P[i][batch_logic][A]))
                        rho_SSE_batch       = np.hstack((rho_SSE_batch, self.rho[i][batch_logic][A]))
                        u_SSE_batch         = np.hstack((u_SSE_batch, self.u[i][batch_logic][A]))
                        v_SSE_batch         = np.hstack((v_SSE_batch, self.v[i][batch_logic][A]))
                        Et_SSE_batch        = np.hstack((Et_SSE_batch, self.Et[i][batch_logic][A]))

                        P_back_PINN_batch    = np.hstack((P_back_PINN_batch, self.P_back[i][batch_logic][A]))
                        x_PINN_batch         = np.hstack((x_PINN_batch, self.x[i][batch_logic][A])) 
                        y_PINN_batch         = np.hstack((y_PINN_batch, self.y[i][batch_logic][A]))
                
                else:
                    true_batch_per_case -= batch_per_case


                #check if the logic 2 (empty node) result in empty array
                if (self.P[i][empty_batch_logic].shape[0] != 0):

                    if  batch_empty_per_case  <= self.P[i][empty_batch_logic].shape[0]:
                        B = np.random.choice(self.P[i][empty_batch_logic].shape[0], size=batch_empty_per_case , replace=False)
                    else:
                        B = np.random.choice(self.P[i][empty_batch_logic].shape[0], size=batch_empty_per_case , replace=True)

                    if(len(B)>0):
                        P_back_PINN_batch    = np.hstack((P_back_PINN_batch, self.P_back[i][empty_batch_logic][B]))
                        x_PINN_batch         = np.hstack((x_PINN_batch, self.x[i][empty_batch_logic][B])) 
                        y_PINN_batch         = np.hstack((y_PINN_batch, self.y[i][empty_batch_logic][B]))

                else:
                    true_batch_per_case -= batch_empty_per_case


            w = np.random.choice(self.P_w.shape[1], size= int(math.ceil(self.bc_ratio*self.x_w.shape[1])) , replace=False)  #10% x 211
            i = np.random.choice(self.P_i.shape[1], size= int(math.ceil(self.bc_ratio*self.x_i.shape[1])) , replace=False)  #10% x 39
            o = np.random.choice(self.P_o.shape[1], size= int(math.ceil(self.bc_ratio*self.x_o.shape[1])) , replace=False)  #10% x 39
            c = np.random.choice(self.P_c.shape[1], size= int(math.ceil(self.bc_ratio*self.x_c.shape[1])) , replace=False)  #10% x 211

            P_back_SSE_batch    = np.hstack((P_back_SSE_batch,self.P_back_w[cases][:,w].flatten(),self.P_back_i[cases][:,i].flatten() ,self.P_back_o[cases][:,o].flatten()   ,self.P_back_c[cases][:,c].flatten())).reshape(-1,1) 
            x_SSE_batch         = np.hstack((x_SSE_batch     ,self.x_w[cases][:,w].flatten()     ,self.x_i[cases][:,i].flatten()      ,self.x_o[cases][:,o].flatten()        ,self.x_c[cases][:,c].flatten())).reshape(-1,1)
            y_SSE_batch         = np.hstack((y_SSE_batch     ,self.y_w[cases][:,w].flatten()     ,self.y_i[cases][:,i].flatten()      ,self.y_o[cases][:,o].flatten()        ,self.y_c[cases][:,c].flatten())).reshape(-1,1)
            P_SSE_batch         = np.hstack((P_SSE_batch     ,self.P_w[cases][:,w].flatten()     ,self.P_i[cases][:,i].flatten()      ,self.P_o[cases][:,o].flatten()        ,self.P_c[cases][:,c].flatten())).reshape(-1,1)
            rho_SSE_batch       = np.hstack((rho_SSE_batch   ,self.rho_w[cases][:,w].flatten()   ,self.rho_i[cases][:,i].flatten()    ,self.rho_o[cases][:,o].flatten()      ,self.rho_c[cases][:,c].flatten())).reshape(-1,1)
            u_SSE_batch         = np.hstack((u_SSE_batch     ,self.u_w[cases][:,w].flatten()     ,self.u_i[cases][:,i].flatten()      ,self.u_o[cases][:,o].flatten()        ,self.u_c[cases][:,c].flatten())).reshape(-1,1)
            v_SSE_batch         = np.hstack((v_SSE_batch     ,self.v_w[cases][:,w].flatten()     ,self.v_i[cases][:,i].flatten()      ,self.v_o[cases][:,o].flatten()        ,self.v_c[cases][:,c].flatten())).reshape(-1,1)
            Et_SSE_batch        = np.hstack((Et_SSE_batch    ,self.Et_w[cases][:,w].flatten()    ,self.Et_i[cases][:,i].flatten()     ,self.Et_o[cases][:,o].flatten()       ,self.Et_c[cases][:,c].flatten())).reshape(-1,1)


            #shape 1D [case size * true batch per case,]
            #convert to shape 2D [case size, true batch per case]
            #transpose shape 2D [batch per case, case size]
            if P_back_PINN_batch.shape[0] != 0:
                P_back_PINN_batch    = np.reshape(P_back_PINN_batch,(-1,true_batch_per_case)).transpose()
                x_PINN_batch         = np.reshape(x_PINN_batch,(-1,true_batch_per_case)).transpose()
                y_PINN_batch         = np.reshape(y_PINN_batch,(-1,true_batch_per_case)).transpose()

            else:
                P_back_PINN_batch    = self.P_back_w[cases][:,w].transpose()        
                x_PINN_batch         = self.x_w[cases][:,w].transpose()
                y_PINN_batch         = self.y_w[cases][:,w].transpose()


            P_back_PINN_batch    = np.vstack((P_back_PINN_batch ,self.P_back_w[cases][:,w].transpose()  ,self.P_back_i[cases][:,i].transpose()  ,self.P_back_o[cases][:,o].transpose()))
            x_PINN_batch         = np.vstack((x_PINN_batch      ,self.x_w[cases][:,w].transpose()       ,self.x_i[cases][:,i].transpose()       ,self.x_o[cases][:,o].transpose()))
            y_PINN_batch         = np.vstack((y_PINN_batch      ,self.y_w[cases][:,w].transpose()       ,self.y_i[cases][:,i].transpose()       ,self.y_o[cases][:,o].transpose()))
            


        return P_back_SSE_batch, x_SSE_batch, y_SSE_batch, P_SSE_batch, rho_SSE_batch, u_SSE_batch, v_SSE_batch, Et_SSE_batch, P_back_PINN_batch, x_PINN_batch, y_PINN_batch 