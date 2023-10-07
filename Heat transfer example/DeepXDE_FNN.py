import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from deepxde.backend import tf
tf.config.optimizer.set_jit(False)
import os
import deepxde as dde
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

m = 101
batch_size = 64
seed = 123 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

dde.config.set_default_float("float64")
Heat_Amp = np.load('/projects/bblp/jaewanp2/Challenging/Heat_transfer_4000_train_GRU/full_data_challenging/flux_amp_all.npy').astype(np.float64)
Temp = np.load('/projects/bblp/jaewanp2/Challenging/Heat_transfer_4000_train_GRU/full_data_challenging/temp.npy').astype(np.float64)
failed_sims = np.load('/projects/bblp/jaewanp2/Challenging/Heat_transfer_4000_train_GRU/full_data_challenging/failed_sims.npy').astype(np.float64)
xy_train_testing = np.load('/projects/bblp/jaewanp2/Challenging/Heat_transfer_4000_train_GRU/full_data_challenging/xy_train_testing.npy').astype(np.float64)
Heat_Amp = Heat_Amp[:4000]

# # Removed failed
# Use_Amp = Heat_Amp[0:int(failed_sims[len(failed_sims)-1]+1)]
# list_failed_sims = np.ndarray.tolist(failed_sims)
# proper_index = []
# for i in range(8000):
# 	pivot = 0
# 	for j in list_failed_sims:
# 		if i == j:
# 			pivot = 1
# 			break
# 	if pivot == 0:
# 		proper_index.append(i)
# Heat_Amp = Use_Amp[proper_index]


for idx , fraction_train in enumerate([ 0.5 , 0.6 , 0.7 , 0.8 ]):
	print('fraction_train = ' + str(fraction_train) )

	# Train / test split
	N_valid_case = len(Heat_Amp)
	N_train = int( N_valid_case * fraction_train )
	train_case = np.random.choice( N_valid_case , N_train , replace=False )
	test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )


	u0_train = Heat_Amp[ train_case , :: ]
	u0_testing = Heat_Amp[ test_case , :: ]
	s_train = Temp[ train_case , : ]
	s_testing = Temp[ test_case , : ]


	print('u0_train.shape = ',u0_train.shape)
	print('type of u0_train = ', type(u0_train))
	print('u0_testing.shape = ',u0_testing.shape)
	print('s_train.shape = ',s_train.shape)
	print('s_testing.shape = ',s_testing.shape)
	print('xy_train_testing.shape', xy_train_testing.shape)

	x_train = (u0_train, xy_train_testing)
	y_train = s_train 
	x_test = (u0_testing, xy_train_testing)
	y_test = s_testing
	data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)


	net = dde.maps.DeepONetCartesianProd(
		[m, 100, 100, 100, 100, 100, 100], [2, 100, 100, 100, 100, 100, 100], "relu", "Glorot normal"
	)

	model = dde.Model(data, net)
	model.compile(
		"adam",
		lr=1e-3,
		decay=("inverse time", 1, 1e-4),
		metrics=["mean l2 relative error"],
	)
	losshistory, train_state = model.train(epochs=350000, batch_size=batch_size, model_save_path="./mdls/TrainFrac_"+str(idx) )
	np.save('losshistory'+str(idx)+'.npy',losshistory)

	import time as TT
	st = TT.time()
	y_pred = model.predict(data.test_x)
	duration = TT.time() - st
	print('y_pred.shape =', y_pred.shape)
	print('Prediction took ' , duration , ' s' )
	print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )
	np.savez_compressed('TestData'+str(idx)+'.npz',a=y_test,b=y_pred,c=u0_testing,d=xy_train_testing)

	error_s = []
	for i in range(len(y_pred)):
		error_s_tmp = np.linalg.norm(y_test[i] - y_pred[i]) / np.linalg.norm(y_test[i])

		if error_s_tmp > 1:
			error_s_tmp = 1
			
		error_s.append(error_s_tmp)
	error_s = np.stack(error_s)
	print("error_s = ", error_s)

	#Calculate mean and std for all testing data samples
	print('mean of relative L2 error of s: {:.2e}'.format(error_s.mean()))
	print('std of relative L2 error of s: {:.2e}'.format(error_s.std()))

	plt.hist( error_s.flatten() , bins=25 )
	plt.savefig('Err_hist_DeepONet'+str(idx)+'.jpg' , dpi=300)