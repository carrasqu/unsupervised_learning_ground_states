import tensorflow as tf
import input_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# lattice definitions. Neighbours, plaquettes, and vertices.
def plqu(lx):
  k=0
  ly=lx
  nh=2*lx*ly
  neig=np.zeros((lx*ly,4),dtype=np.int)
  for j in range(ly):
    for i in range(lx):
       if i<lx-1:
         neig[k,0]=k+1
         if j<ly-1:
          neig[k,1]=k+lx
         elif j==ly-1:
          neig[k,1]=k-(ly-1)*lx
         if i==0:
           neig[k,2]=k+lx-1
         else:
           neig[k,2]=k-1
       elif i==lx-1:
         neig[k,0]=k-(lx-1)
         if j<ly-1:
          neig[k,1]=k+lx
         elif j==ly-1:
          neig[k,1]=k-(ly-1)*lx
         neig[k,2]=k-1
       if j==0:
         neig[k,3]=k+(ly-1)*lx
       else:
         neig[k,3]=k-lx
       k=k+1

  plaquette=np.zeros((lx*ly,4),dtype=np.int)
  vertex=np.zeros((lx*ly,4),dtype=np.int)
  for i in range(ly*lx):
    plaquette[i,0]=2*i
    plaquette[i,1]=2*i+1
    plaquette[i,2]=2*neig[i,0]+1
    plaquette[i,3]=2*neig[i,1]
    vertex[i,0]=2*i
    vertex[i,1]=2*i+1
    vertex[i,2]=2*neig[i,2]
    vertex[i,3]=2*neig[i,3]+1
    #print "p", i, plaquette[i,0],  plaquette[i,1], plaquette[i,2], plaquette[i,3]
    #print "v", i, vertex[i,0], vertex[i,1], vertex[i,2], vertex[i,3]
  return neig,plaquette,vertex

# defining weighs and initlizatinon
def weight_variable(shape):
  #initial = tf.random_normal(shape, stddev=0.05)
  initial = tf.truncated_normal(shape, stddev=1)  
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(-0.05, shape=shape)
  return tf.Variable(initial)

# defining the convolutional and max pool layers
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def x_to_xconv(lx,ly,neig,spinreal):
        spin=np.zeros((lx+1)*(ly+1)*2,dtype=np.int)
        nh=lx*ly*2
        xc=[np.empty( shape=(0),dtype=np.int )]
        for i in range(nh-1): 
            a=np.empty( shape=(0),dtype=np.int )
            xc.append(a)
        #xc=np.asarray(xc)  
            
        nh_new=(lx+1)*(ly+1)*2
        flipthis = [0] * nh
	k=0
	kn=0
	for j in range(ly+1):
	 if j==ly:
	     k=0
	 for i in range(lx+1):
	
	    if i<lx:
	     #print kn+1,k+1
	     #xc[:,kn]=test[:,k]
             #xc[k]=kn
             spin[kn]=spinreal[k]
             xc[k]=np.append(xc[k],kn)      
	     kn=kn+1
	     k=k+1
	     #print kn+1,k+1
	     #xc[:,kn]=test[:,k]
             #xc[k]=kn
             spin[kn]=spinreal[k]   
             xc[k]=np.append(xc[k],kn)  
	     kn=kn+1
	     k=k+1
	    elif i==lx:
	     #print "neighs"
	     #print kn+1,2*neig[(k-1)/2,0] +1
	     #xc[:,kn]=test[:,2*neig[(k-1)/2,0]]
             #xc[k]=kn
             spin[kn]=spinreal[2*neig[(k-1)/2,0]] 
             xc[2*neig[(k-1)/2,0]]=np.append(xc[2*neig[(k-1)/2,0]],kn)   
	     kn=kn+1
	     #print kn+1,2*neig[(k-1)/2,0]+1+1
	     #xc[:,kn]=test[:,2*neig[(k-1)/2,0]+1]
             #xc[k]=kn
             spin[kn]=spinreal[2*neig[(k-1)/2,0]+1] 
             xc[2*neig[(k-1)/2,0]+1 ]=np.append(xc[2*neig[(k-1)/2,0]+1],kn)
	     kn=kn+1
         
        xc=np.asarray(xc)
        return xc,spin

def getp(v,plaquette,xc):
    pp=np.zeros(4,dtype=np.int)
    for i in range(4):
      pp[i]=xc[plaquette[v]][i][0]     
    return pp
def flips(who,xc,x): # creates a new configuration xnew flipping on the "real" lattice sites contained in the vector who
     xnew=np.copy(x)
     xnew[:,np.hstack(xc[who].flat)]=-xnew[:,np.hstack(xc[who].flat)]
     return xnew


# defining the geometry
lx=2
nh=(lx)*(lx)*2
nh_conv=(lx+1)*(lx+1)*2
neig,plaquette,vertex=plqu(lx) # plaquettes, vertices and neighbours
spin=2*np.random.randint(2, size=nh)-1
xc,spin=x_to_xconv(lx,lx,neig,spin) # the configurations contain the boundary condition, so when flipping spins we need a table to map back to those configurations. 
spin=np.reshape(spin,(1,nh_conv))

# Hamiltonian parameters of the toric code in a field
Jv=1.0
Jp=1.0
hx=0.0


# ###############defining the wave function: convolutional neural net #############


# defining the model
numberlabels=1
x = tf.placeholder("float", shape=[None, (lx+1)*(lx+1)*2]) # placeholder for the spin configurations
#x = tf.placeholder("float", shape=[None, lx*lx*2]) #with padding and no PBC conv net
y_ = tf.placeholder("float", shape=[None, numberlabels])
#first layer 
# convolutional layer # 2x2 patch size, 2 channel (2 color), 64 feature maps computed
nmaps1=64
W_conv1 = weight_variable([2, 2, 2,nmaps1])
# bias for each of the feature maps
b_conv1 = bias_variable([nmaps1])
# applying a reshape of the data to get the two dimensional structure back
#x_image = tf.reshape(x, [-1,lx,lx,2]) # #with padding and no PBC conv net
x_image = tf.reshape(x, [-1,(lx+1),(lx+1),2]) # with PBC 
#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1) # removing the maxpool layer
h_pool1=h_conv1
#In order to build a deep network, we stack several layers of this type. The second layer will have 8 features for each 5x5 patch. 
# weights and bias of the fully connected (fc) layer. Ihn this case everything looks one dimensiona because it is fully connected
nmaps2=64
#W_fc1 = weight_variable([(lx/2) * (lx/2) * nmaps1,nmaps2 ]) # with maxpool
W_fc1 = weight_variable([(lx) * (lx) * nmaps1,nmaps2 ]) # no maxpool images remain the same size after conv
b_fc1 = bias_variable([nmaps2])
# first we reshape the outcome h_pool2 to a vector
#h_pool1_flat = tf.reshape(h_pool1, [-1, (lx/2)*(lx/2)*nmaps1]) # with maxpool
h_pool1_flat = tf.reshape(h_pool1, [-1, (lx)*(lx)*nmaps1]) # no maxpool
# then apply the ReLU with the fully connected weights and biases.
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
# readout layer. Finally, we add a softmax layer, just like for the one layer softmax regression above.
# weights and bias
W_fc2 = weight_variable([nmaps2, numberlabels])
b_fc2 = bias_variable([numberlabels])
# apply a softmax layer
#y_conv=tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
y_conv=(tf.matmul(h_fc1, W_fc2) + b_fc2)

# log of the wave function used in the logarithmic derivative
logy=tf.log(y_conv)
grad=tf.gradients(logy,[W_conv1,b_conv1,W_fc1,b_fc1,W_fc2,b_fc2])


# Yconv is the amplitud of the wave function in the sz basis 
#####################################################################

# initializing the tensorflow stuff
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#initializing the wave function
psi = sess.run(y_conv,feed_dict={ x:spin})

#thermalization
ntherm=5
nstep=100
nbin_t=50
gamma=0.2
print "thermalization.."
for i in range(ntherm):
 for j in range(nstep):

    # generating the "training set" via metropolis algorithm
    for k in range(nh):
        #GaugeUpdate
        ii=np.random.randint(vertex.shape[0], size=1)
        spint=flips(vertex[ii,:],xc,spin)
        psinew= sess.run(y_conv,feed_dict={ x:spint})
        r=np.random.rand(1)
        #print "gauge"
        #print r, psinew[0][0], psi[0][0], np.power(psinew[0][0]/psi[0][0],2)
        #print "spin",spin
        #print "spint",spint
        #print "diff",spin-spint 
        #print "old wfs", psi,psinew    
        #print vertex[ii,:]
        #print r, psinew[0][0], psi[0][0], np.power(psinew[0][0]/psi[0][0],2)
        if np.power(psinew[0][0]/psi[0][0],2)> r :
           spin=np.copy(spint)
           psi=np.copy(psinew)
           #print "accepted spin",spin
           #print "accepted wf", psi
           #sys.exit()
           #print "accepted gauge"
        #LocalUdate
        ii=np.random.randint(nh,size=1)
        spint=flips([ii],xc,spin)
        psinew= sess.run(y_conv,feed_dict={ x:spint})
        r=np.random.rand(1)
        #print "local" 
        #print r, psinew[0][0], psi[0][0], np.power(psinew[0][0]/psi[0][0],2) 
        if np.power(psinew[0][0]/psi[0][0],2)> r :
           spin=np.copy(spint)
           psi=np.copy(psinew)
           #print "accepted local"

print spin
print "thermalization ready"
nbin=1000
nstep=10000
nbin_t=100
for i in range(nbin):
 acc=0
 accg=0
 for j in range(nstep):
   
    # generating the "training set" via metropolis algorithm 
    for k in range(nh): 
        #GaugeUpdate
        ii=np.random.randint(vertex.shape[0], size=1)
        spint=flips(vertex[ii,:],xc,spin)
        psinew= sess.run(y_conv,feed_dict={ x:spint})
        r=np.random.rand(1)
        if np.power(psinew[0][0]/psi[0][0],2)> r :
           spin=np.copy(spint)
           psi=np.copy(psinew) 
           accg=accg+1 
           #print "accepted gauge"  
        #LocalUdate
        #ii=np.random.randint(nh,size=1)
        spint=flips([ii],xc,spin)
        psinew= sess.run(y_conv,feed_dict={ x:spint})
        r=np.random.rand(1)
        if np.power(psinew[0][0]/psi[0][0],2)> r :
           spin=np.copy(spint)
           psi=np.copy(psinew)
           acc=acc+1
           #print "accepted local"
     
    #accumulate training
    if j==nbin_t: 
       tset=np.copy(spin[0,:])
       wfset=np.copy(psi)
    if j>nbin_t:
       #tset=np.append(tset,spin[0,:],axis=0) 
       tset=np.vstack((tset, spin[0,:]))
       wfset=np.vstack((wfset, psi)) 
  
 # Computing the observables necessary for the gradient of the energy
 ###local energy ex##
 ex=np.zeros((tset.shape[0],1))
 exdiag=np.zeros((tset.shape[0],1))
 for v in range(lx*lx):
     #plaquette term sxsxsxsx:
     fset=flips(vertex[v,:],xc,tset)  
     yp=sess.run(y_conv,feed_dict={ x:fset}) 
     ex=ex-Jv*yp
     pp=getp(v,plaquette,xc)  
     exdiag=exdiag-Jp*np.reshape(np.prod(tset[:,pp],axis=-1 ),(tset.shape[0],1)) 
 ex=np.divide(ex,wfset) # off-diagonal local energy ex computed
 #diagonal part
 ex=ex+exdiag
 #print ex 
 #print ex/(lx*lx*2) 
 np.savetxt('ex.txt'+str(i), ex/(lx*lx*2))  
 np.savetxt('wf.txt'+str(i),wfset) 
 #print "single spin fliop acceptance",(acc+0.00)/(nstep*nh)
 #print "gauge update acceptance",(accg+0.0)/(nstep*nh) 
 energy=np.sum(ex)/tset.shape[0]

 #computing gradientes per example the shitty way
 
 iii=0
 grads=sess.run(grad,feed_dict={ x:np.reshape(tset[iii,:],(1,tset.shape[1]))})
 #print np.asarray(grads)
 DW=2*np.asscalar(ex[iii])*np.asarray(grads)-2*energy*np.asarray(grads)
 for iii in range(1,tset.shape[0]):
     grads=sess.run(grad,feed_dict={ x:np.reshape(tset[iii,:],(1,tset.shape[1]))}) 
     #print np.asarray(grads)[0] 
     DW=DW+2*np.asscalar(ex[iii])*np.asarray(grads)-2*energy*np.asarray(grads) 
      
  
 DW=DW/tset.shape[0]
 #print DW 

 Wc1=sess.run(W_conv1)
 bc1=sess.run(b_conv1)
 wfc1=sess.run(W_fc1)
 bfc1=sess.run(b_fc1)
 wfc2=sess.run(W_fc2)
 bfc2=sess.run(b_fc2)
 
 Wc1=Wc1-gamma*DW[0]
 bc1=bc1-gamma*DW[1]
 wfc1=wfc1-gamma*DW[2]
 bfc1=bfc1-gamma*DW[3] 
 wfc2=wfc2-gamma*DW[4]
 bfc2=bfc2-gamma*DW[5]


 aW_conv1 = W_conv1.assign(Wc1)
 sess.run(aW_conv1) 
 ab_conv1 = b_conv1.assign(bc1)
 sess.run(ab_conv1)
 aW_fc1 = W_fc1.assign(wfc1)
 sess.run(aW_fc1)
 ab_fc1 = b_fc1.assign(bfc1)
 sess.run(ab_fc1)
 aW_fc2 = W_fc2.assign(wfc2)
 sess.run(aW_fc2)
 ab_fc2 = b_fc2.assign(bfc2)
 sess.run(ab_fc2) 
 
 #W_conv1 = tf.convert_to_tensor(Wc1, dtype=tf.float32) 
 #b_conv1=tf.convert_to_tensor(bc1, dtype=tf.float32)    
 #W_fc1 = tf.convert_to_tensor(wfc1, dtype=tf.float32)
 #b_fc1=tf.convert_to_tensor(bfc1, dtype=tf.float32) 
 #W_fc2 = tf.convert_to_tensor(wfc2, dtype=tf.float32)
 #b_fc2=tf.convert_to_tensor(bfc2, dtype=tf.float32)
 #sess.run(tf.initialize_all_variables())
 
 
  
 
 # the energy
 print energy/(lx*lx*2),(acc+0.00)/(nstep*nh),(accg+0.00)/(nstep*nh) 
 
 # the logarithmic gradient
   
