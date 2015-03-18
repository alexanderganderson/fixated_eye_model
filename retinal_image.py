from EM_burak_class import *

emb = EMBurak(_DC = 0.1)
emb.gen_data()

R = emb.R

L_N = emb.L_N

#plt.imshow(R[:, 0].reshape(L_N, L_N), 
#           interpolation = 'nearest', cmap = plt.cm.gray_r)
#plt.show()

mean = np.mean(R, axis = 1)
plt.imshow(mean.reshape(L_N, L_N), 
           interpolation = 'nearest', cmap = plt.cm.gray_r)
plt.show()
           