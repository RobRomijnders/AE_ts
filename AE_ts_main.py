# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:43:29 2016

@author: Rob Romijnders
"""
direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'


import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


import matplotlib.pyplot as plt
import os
from tensorflow.contrib.tensorboard.plugins import projector
from AE_ts_model import Model, open_data, plot_data, plot_z_run


"""Hyperparameters"""
LOG_DIR = "/home/rob/Dropbox/ml_projects/AE_ts/log_tb"
config = {}                             #Put all configuration information into the dict
config['num_layers'] = 2                #number of layers of stacked RNN's
config['hidden_size'] = 90              #memory cells in a layer
config['max_grad_norm'] = 5             #maximum gradient norm during training
config['batch_size'] = batch_size = 64  
config['learning_rate'] = .005
config['crd'] = 1                       #Hyperparameter for future generalization
config['num_l'] = 20                    #number of units in the latent space

plot_every = 100                        #after _plot_every_ GD steps, there's console output
max_iterations = 1000                   #maximum number of iterations
dropout = 0.8                           #Dropout rate 
"""Load the data"""
X_train,X_val,y_train,y_val = open_data('/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015')
  
N = X_train.shape[0]
Nval = X_val.shape[0]
D = X_train.shape[1]
config['sl'] = sl = D          #sequence length
print('We have %s observations with %s dimensions'%(N,D))


# Organize the classes
num_classes = len(np.unique(y_train))
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
  y_train -=base
  y_val -= base

#Plot data   # and save high quality plt.savefig('data_examples.eps', format='eps', dpi=1000)
plot_data(X_train,y_train)


#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))


"""Training time!"""
model = Model(config)
sess = tf.Session()
perf_collect = np.zeros((2,int(np.floor(max_iterations/plot_every))))

if True:
  sess.run(model.init_op)
  writer = tf.summary.FileWriter(LOG_DIR, sess.graph)  #writer for Tensorboard

  step = 0      # Step is a counter for filling the numpy array perf_collect
  for i in range(max_iterations):
    batch_ind = np.random.choice(N,batch_size,replace=False)
    result = sess.run([model.loss, model.loss_seq,model.loss_lat_batch,model.train_step],feed_dict={model.x:X_train[batch_ind],model.keep_prob:dropout})
    
    if i%plot_every == 0:
      #Save train performances
      perf_collect[0,step] = loss_train = result[0]
      loss_train_seq, lost_train_lat = result[1], result[2]

      #Calculate and save validation performance
      batch_ind_val = np.random.choice(Nval,batch_size,replace=False)

      result = sess.run([model.loss, model.loss_seq,model.loss_lat_batch,model.merged], feed_dict={ model.x: X_val[batch_ind_val],model.keep_prob:1.0})
      perf_collect[1,step] = loss_val = result[0]
      loss_val_seq, lost_val_lat = result[1], result[2]
      #and save to Tensorboard
      summary_str = result[3]
      writer.add_summary(summary_str, i)
      writer.flush()
      
      print("At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" %(i,max_iterations,loss_train,loss_train_seq, lost_train_lat,loss_val, loss_val_seq, lost_val_lat))
      step +=1
if False:
  ##Extract the latent space coordinates of the validation set
  start = 0
  label = []   #The label to save to visualize the latent space
  z_run = []

  while start + batch_size < Nval:
    run_ind = range(start,start+batch_size)
    z_mu_fetch = sess.run(model.z_mu, feed_dict = {model.x:X_val[run_ind],model.keep_prob:1.0})
    z_run.append(z_mu_fetch)
    start += batch_size

  z_run = np.concatenate(z_run,axis=0)
  label = y_val[:start]
  
  plot_z_run(z_run,label)



#Save the projections also to Tensorboard
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step)
config = projector.ProjectorConfig()
# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = model.z_mu.name 
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Saves a configuration file that TensorBoard will read during startup.
projector.visualize_embeddings(writer, config)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), step+1)
writer.flush()



#Now open Tensorboard with
#  $tensorboard --logdir = LOG_DIR