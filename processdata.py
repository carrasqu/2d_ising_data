import numpy as np
import sys 

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
  return neig


lx = 6
Nq = lx*lx 
ivic = plqu(lx)



Nmodel = 50

S=0



for i in range(S,Nmodel,1):
    x = np.loadtxt("samplex_"+str(i)+".txt")
    x = 2*x-3
    E = 0
    E2 = 0
    m = 0
    m2 = 0
    for k in range(x.shape[0]):
        ee=0.0
        #print x[i,:]   
        for j in range(Nq): 
            ee+=-x[k,j]*(x[k,ivic[j,0]] + x[k,ivic[j,1]])
        
        E += ee
        E2+= ee**2
        mm = np.abs(np.sum(x[k,:])) 
        m += mm/Nq
        m2 += mm**2 
        
    E  = E/x.shape[0]
    E2 = E2/x.shape[0]
    E2 = np.sqrt(np.abs( E2 - E**2)/x.shape[0])
    m = m/x.shape[0]
    m2 = m2/x.shape[0]  
    m2 = np.sqrt(np.abs( m2 - m**2)/x.shape[0])
    
    #m = np.mean(x)
      
    #em = np.mean(np.sum(x,axis=1)**2)
    #em = np.sqrt(np.abs(m**2-em )/x.shape[0]) 
    #E  = -np.mean(np.sum(x[:,0:-1]* x[:,1:],axis=1)+x[:,0]*x[:,-1])/x.shape[1]
    #eE = np.mean((np.sum(x[:,0:-1]* x[:,1:],axis=1)+x[:,0]*x[:,-1])**2)
    #eE = np.sqrt(np.abs(E**2-eE )/x.shape[0])


    print i, E/Nq,E2/Nq, m,m2 # , exact[3], exact[5] 
    


