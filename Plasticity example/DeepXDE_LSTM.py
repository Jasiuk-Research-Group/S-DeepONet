import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from deepxde.backend import tf
tf.config.optimizer.set_jit(False)
import os
import deepxde as dde
dde.config.disable_xla_jit()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

m = 101
batch_size = 64
seed = 123 
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(seed)

dde.config.set_default_float("float64")
u0_train = np.load('/scratch/bblv/skoric/DEVELOP_DOGBONE_REFINED_SCALED_AMP/N15K_L6_NE_100/data_amp_train.npy').astype(np.float64)
u0_train = np.expand_dims( u0_train , -1 )
u0_testing = np.load('/scratch/bblv/skoric/DEVELOP_DOGBONE_REFINED_SCALED_AMP/N15K_L6_NE_100/data_amp_testing.npy').astype(np.float64)
u0_testing = np.expand_dims( u0_testing , -1 )
u0_all = np.concatenate([u0_train,u0_testing],axis=0)

s_train = np.load('/scratch/bblv/skoric/DEVELOP_DOGBONE_REFINED_SCALED_AMP/N15K_L6_NE_100/s_train.npy').astype(np.float64)
s_testing = np.load('/scratch/bblv/skoric/DEVELOP_DOGBONE_REFINED_SCALED_AMP/N15K_L6_NE_100/s_testing.npy').astype(np.float64)
s_all = np.concatenate([s_train,s_testing],axis=0)

xy_train_testing = np.load('/scratch/bblv/skoric/DEEPXDE_DEEPONET_PLASTICITY/SIX_CONT_POINTS_10K_DATA/xy_train_testing.npy').astype(np.float64)



for idx , fraction_train in enumerate([ 0.5 , 0.6 , 0.7 , 0.8 ]):
	print('fraction_train = ' + str(fraction_train) )

	# Train / test split
	N_valid_case = len(u0_all)
	N_train = int( N_valid_case * fraction_train )
	train_case = np.random.choice( N_valid_case , N_train , replace=False )
	test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )


	u0_train = u0_all[ train_case , :: ]
	u0_testing = u0_all[ test_case , :: ]
	s_train = s_all[ train_case , : ]
	s_testing = s_all[ test_case , : ]


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


	activation = "relu"
	branch = tf.keras.models.Sequential([
		 tf.keras.layers.InputLayer(input_shape=(m,1)),
		 tf.keras.layers.LSTM(units=256,activation = 'tanh',return_sequences = True),
		 tf.keras.layers.LSTM(units=128,activation = 'tanh',return_sequences = False),
		 tf.keras.layers.RepeatVector(m),
		 tf.keras.layers.LSTM(units=128,activation = 'tanh',return_sequences = True),
		 tf.keras.layers.LSTM(units=256,activation='tanh',return_sequences = True ),
		 tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
		 tf.keras.layers.Reshape((101,))
		 ])

	net = dde.maps.DeepONetCartesianProd(
			[m, branch], [2, 101, 101, 101, 101, 101, 101], activation, "Glorot normal")


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