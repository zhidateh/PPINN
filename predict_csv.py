import utils as utils
import numpy as np
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case', help='Case ID')
parser.add_argument('region', help='Domain/Inlet/Outlet/Centerline/Wall')
parser.add_argument('plot', help='1 for loss, 2 for centerline, 3 for contour')
parser.add_argument('save', help='1 for save, 2 for not save')

args = parser.parse_args()
if(args.save == "1"):
    save = True
else:
    save = False

project_name = args.case 
model_path = os.getcwd() + '/model/%s/'%project_name

for f in os.listdir(model_path):
    if (f.endswith('.csv')):
        pb = (f[f.index('=')+1:f.index('=')+6])

test_path       = os.getcwd() + '/data/%s/bp=%s.csv'%(args.region, pb)
predict_path    = model_path + '%s_bp=%s.csv' %(project_name,pb)

P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,Et_test = utils.loadCSV(test_path)
P_back_pred,x_pred,y_pred,P_pred,rho_pred,u_pred,v_pred,Et_pred = utils.loadCSV(predict_path)


#Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_Et = np.linalg.norm(Et_test- Et_pred,2)/np.linalg.norm(Et_test,2)
print("Test Error in E: "+str(error_Et))


print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%(error_P,error_rho,error_u,error_v,error_Et))
print("%.3f"%((error_P+error_rho+error_u+error_v+error_Et)/5))



if(args.plot == "1"):
    utils.plotLoss( save,
                    model_path + '%s_bp=%s'%(project_name,pb), 
                    0,
                    600,
                    0,
                    10)

elif(args.plot == "2"):
    utils.plotAll_1D(   save,
                        x_test,
                        y_test,
                        P_test,
                        rho_test,
                        u_test,
                        v_test,
                        Et_test,
                        P_pred,
                        rho_pred,
                        u_pred,
                        v_pred,
                        Et_pred)

elif(args.plot == "3"):
    utils.plotAll(  save,
                    x_test,
                    y_test,
                    P_test,
                    rho_test,
                    u_test,
                    v_test,
                    Et_test,
                    P_pred,
                    rho_pred,
                    u_pred,
                    v_pred,
                    Et_pred)
