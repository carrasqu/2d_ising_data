import tensorflow as tf
import numpy as np
import os
import itertools as it
import sys



class LatentAttention():
    def __init__(self, data, K=4, Number_qubits=6 , latent_rep_size=100, gru_hidden=100,decoder='TimeDistributed_mol', Nsamples=0):

        
        """ Class to learn a recurrent neural net model using maximum likelihood estimation
            decoder='TimeDistributed_mol' instead can be used during training using MLE. The input
            at each time step concatenates a trainable variable z to the symbol [z,symbol_t]; 
            during generation input should be [z,predicted_symbol_t]
        """

        # This creates some folders where things are stored 
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if not os.path.exists("training"):
            os.makedirs("training")
        if not os.path.exists("training/train"):
            os.makedirs("training/train") 
        if Nsamples > 0: 
            if not os.path.exists("samples"):
                os.makedirs("samples")      

        self.batchsize = 500  
        self.latent_rep_size = latent_rep_size # used for VAE and for MLE. In the MLE case this is used to define a
        # trainable variable
          
        self.gru_hidden = gru_hidden # size of the hidden state of the GRU 
        self.Nsamples = Nsamples     # how many samples to obtain from the RNN  
       
        # # loading data sets
        self.data_train = np.loadtxt(data)
        
        self.charset_length = K  # Size of the local state space at each time step t. eg Ising model = 2
        self.max_length = Number_qubits  # Size of the sequence. For Ising model in 2d = LxL, L=linear size
        self.n_samples = self.data_train.shape[0] # size of the training set
        self.POVM_meas = tf.placeholder(tf.float32, [None, self.max_length*self.charset_length])
        # self.POVM_meas = placeholder for the input data in a one-hot vector format.
        # Its size is NONE by the number of elements LxL*charset_length
        self.molecules = tf.reshape(self.POVM_meas, [tf.shape(self.POVM_meas)[0], self.max_length, self.charset_length]) 
        # molecules is basically a reshaped version of POVM_meas that is easier to manipulate  

        self.generated_molecules,_ = self.generation(decoder, self.molecules)
        # Generation does two things: If you give it a set of configurations it will compute the probability
        # of the samples in molecues that allows you to compute the cost function below. If you don't give it
        # anything it will give you samples of the RNN as in the next line of this code

        self.sample_onehot, self.logP = self.generation(decoder) # gives you samples X and log(P(X))
        self.sample_RNN = tf.argmax(self.sample_onehot, axis=2) # samples not in one-hot but index format
        
        self.generation_loss = -tf.reduce_sum(self.molecules * tf.log(1e-10 + self.generated_molecules), [1, 2])
        # MLE cost function, the usual thing for each element in the batch
        
        self.cost = tf.reduce_mean(self.generation_loss)  # average over the batch of the cost function 
  
        tf.summary.scalar('Cost_function', self.cost)
    
        # The minimization of the cost function with gradient clipping
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost) # optimizer
            self.opt = tf.train.AdamOptimizer(0.001)
            self.gradients, self.variables = zip(*self.opt.compute_gradients(self.cost))
            for grad, var in zip(self.gradients,self.variables):
                tf.summary.histogram(var.name + '/gradient', grad) # this is to produce a histogram of the gradients

            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)
            self.optimizer = self.opt.apply_gradients(zip(self.gradients, self.variables))   

    def generation(self,decoder, molecules=None):
        if decoder == 'TimeDistributed_mol':
            if molecules == None: # then the RNN samples a configuration
                with tf.variable_scope("generation", reuse=True):
                    zt = tf.get_variable('z', shape=[1, self.latent_rep_size]) # trainable variable
                    z = tf.tile(zt, [self.batchsize, 1]) # tile at each time step
                    # fully connected layer applied to latent vector z
                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu,name="fc_GEN")
#
                # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]);
                y0 = tf.zeros([self.batchsize, self.charset_length], dtype=tf.float32)

                logP = tf.zeros([self.batchsize], dtype=tf.float32) # This is the logP of the sampled configuration

                # concatenates z_matrix to the atom of zeros as the first input of the RNN
                y_z = tf.concat([y0, z_matrix], axis=1)

                # initial state of the RNN during the unrolling
                s0 = tf.zeros([self.batchsize, 3*self.gru_hidden], dtype=tf.float32)

                #output tensor unrolling
                h2 = tf.zeros([self.batchsize, 1, self.charset_length], dtype=tf.float32)

                i0 = tf.constant(0)  # counter

                # preparing the unrolling loop
                # condition
                time_steps = self.max_length

                def c(i, s0, h2, y_z,logP):
                    return i < time_steps
#
                # body of what happens in the unrolling loop

                def b(i, s0, h2, y_z,logP):
                    with tf.variable_scope("generation", reuse=True):
                        # GRU Recurrent NN
                        with tf.variable_scope("rnn"):
                            # cell=tf.contrib.rnn.GRUCell(self.gru_hidden) # GRU cell
                            # cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=False) # stacking 3 GRUs
                            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.gru_hidden) for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs
                            outputs, state = cell(y_z, s0)

                        # apply a softmax layer to the output of the RNN; dlayer is probability accross the different symbols in the dictionary
                        dlayer = tf.layers.dense(outputs,units=self.charset_length,activation=tf.nn.softmax, name="output") # returns probability

                        # Two options. 1 take the symbol with the largest probability using argmax or sample the symbols P(symbol| [sampled symbol i, z],[sampled symbol i-1, z].... [sampled symbol 0, z])
                        # pick the most likely symbol as input for the next loop step
                        # amax=tf.argmax( dlayer,axis=1 )
                        # dlayer=tf.one_hot(amax,depth=self.charset_length,axis=1, dtype=tf.float32)

                        ## samples dlayer probabilities
                        logits  = tf.log(dlayer)
                         
                        samples = tf.reshape(tf.multinomial( logits,1),[-1,]) # reshape to a vector of shape=(self.batchsize,)l  multinomial returns one integer sample per element in the batch
                        dlayer  = tf.one_hot(samples,depth=self.charset_length,axis=1, dtype=tf.float32) # onehot symbol
                        
                        logP = logP + tf.reduce_sum(dlayer*logits,[1])
  
                    return [i+1,state,tf.concat([h2,tf.reshape(dlayer,[self.batchsize,1, self.charset_length])],axis=1),tf.concat([dlayer, z_matrix],axis=1),logP]

                # unrolling using tf.while_loop
                ii, s0, h2, y_z,logP = tf.while_loop(
                        c, b, loop_vars=[i0, s0, h2, y_z,logP],
                        shape_invariants=[i0.get_shape(), s0.get_shape(), tf.TensorShape(
                            [self.batchsize, None, self.charset_length]), y_z.get_shape(),logP.get_shape()])  # shape invariants required
                # since h2 increases size as rnn unrolls

                h2=tf.slice(h2, [0, 1, 0], [-1, -1, -1])  # cuts the initial zeros that inserted to start the while_loop
                
            else:
                 
                with tf.variable_scope("generation", reuse=None):

                    # #fully connected layer applied to latent vector z
                    zt =  tf.get_variable('z', shape=[1, self.latent_rep_size])
                    z = tf.tile(zt, [self.batchsize, 1])
  
                    z_matrix = tf.layers.dense(inputs=z, units=self.latent_rep_size, activation=tf.nn.relu, name="fc_GEN")

                    # An atom of zeros. This is part of the (t=0) input of the RNN: input is concat([zmol, z_matrix]); inputs at time t=1,2,.. maxlength-1  are concat([molecule[:,t,:], z_matrix])
                    zmol = tf.zeros([self.batchsize, 1, self.charset_length], dtype=tf.float32)

                    # creates the new "molecule" input, i.e., zmol,mol_1,mol_2,...,mol_{maxlength-1}
                    mol = tf.slice(tf.concat([zmol, molecules], axis=1), [0, 0, 0], [self.batchsize, self.max_length,
                                                                                     self.charset_length])  # creates
                    # new "molecule" input, i.e., zmol,mol_1,mol_2,...,mol_{maxlength-1}

                    # concatenates z_matrix to all the different parts of the molecule to create the extended input of
                    # the RNN
                    h1 = tf.stack([tf.concat([t, z_matrix], axis=1) for t in tf.unstack(mol, axis=1)], axis=1)

                    # GRU Recurrent NN
                    #cell = tf.contrib.rnn.GRUCell(self.gru_hidden)  # GRU cell
                    #cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs
                    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.gru_hidden) for _ in range(3)], state_is_tuple=False)  # stacking 3 GRUs
   
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=h1, dtype=tf.float32)  # cells to the input
                    # applying a fully connected layer with a softmax non linearity on top to each time output separately
                    in_dense = tf.reshape(outputs, [self.batchsize*self.max_length, self.gru_hidden])
                    dlayer = tf.layers.dense(in_dense, units=self.charset_length, activation=tf.nn.softmax, name="output")
                    h2 = tf.reshape(dlayer, [self.batchsize, self.max_length, self.charset_length])
                    logP = tf.zeros([self.batchsize], dtype=tf.float32)
#
#                     # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generation'):
#                     # print i
#
#
        return h2,logP
#
#
    def train(self):

        # train
        bcount = 0
        ept = np.random.permutation(self.data_train) # random permutation of the dataset
        saver = tf.train.Saver(max_to_keep=2) # save models
         
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # create a summary writer for tensorboard.
            writer = tf.summary.FileWriter('./logs', graph=sess.graph)
            summary_op = tf.summary.merge_all()


            nstepspe = int(self.n_samples / self.batchsize) # number of gradient updates per epoch

            rlist=range(int(self.n_samples / self.batchsize))
            lastNumber = rlist[-1]

            
            
            counter=0
            for epoch in range(50): # run optimization for 50 epochs
                for idx in range(nstepspe):
                    if bcount*self.batchsize+ self.batchsize>=self.n_samples:
                        bcount=0
                        ept=np.random.permutation(self.data_train) # pick a random permutation of the dataset

                    batch=ept[ bcount*self.batchsize: bcount*self.batchsize+self.batchsize,:] # pick a batch
                    bcount=bcount+1


                    _, gen_loss = sess.run((self.optimizer,  self.generation_loss),
                                            feed_dict={self.POVM_meas: batch }) # runs optimizer and returns loss
                    counter = counter+1
                      
                     # print cost every at the end of each epoch
                    if idx ==  lastNumber:

                        print "epoch  loss " , epoch, np.mean(gen_loss)
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                
                        # Obtaining samples from the model at epoch 
                        if self.Nsamples != 0:
                            Ncalls = self.Nsamples /self.batchsize
                            samples,lP = sess.run([self.sample_RNN,self.logP])
                            lP =np.reshape(lP,[-1,1])
 
 
                            for k in range(Ncalls): 
                                sa,llpp = sess.run([self.sample_RNN,self.logP])
                                llpp =np.reshape(llpp,[-1,1])  
                                samples = np.vstack((samples,sa))
                                lP =  np.vstack((lP,llpp))
 
                                 
                            np.savetxt('./samples/samplex_'+str(epoch)+'.txt',samples+1,fmt='%i') # saves samples
                            np.savetxt('./samples/logP_'+str(epoch)+'.txt',lP) # saves the logP of the samples above

         
           
            

# define the model
model = LatentAttention(data='./data/train.txt', K=2, Number_qubits = int(sys.argv[1]), latent_rep_size=int(sys.argv[2]), gru_hidden=int(sys.argv[3]), decoder='TimeDistributed_mol', Nsamples = int(sys.argv[4]) )
# Train the model
model.train()
