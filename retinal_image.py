from EM_burak_class import *
from scipy.signal import convolve2d


emb = EMBurak(_DT = 0.003, _DC = 50.)
emb.gen_data()
emb.calculate_inner_products()


R = emb.R

L_N = emb.L_N
N_N = emb.N_N

a = 20.
F = np.exp(- np.arange(0, 3 * a + 1) / (3 * a)) 
F = F / np.sum(F)


R1 = np.zeros_like(R)

for i in range(R1.shape[0]):
    R1[i] = np.convolve(R[i], F, mode = 'same')




t = 50

plt.figure().suptitle('Diffusion constant ' + str(emb.DC) + '\n' +
                        'Lambda0,Lambda1 = (' + str(emb.L0) + ',' + str(emb.L1) + ')'
                        )
plt.subplot(2, 2, 1)
plt.imshow(emb.Ips[0, :, t].reshape(L_N, L_N), 
           interpolation = 'nearest', cmap = plt.cm.gray_r)
plt.title('Retinal Projection of the Image')

plt.subplot(2, 2, 2)
plt.imshow(R[:, t].reshape(L_N, L_N),
            interpolation = 'nearest', cmap = plt.cm.gray_r)
plt.title('Spikes')

plt.subplot(2, 2, 3)
plt.imshow(R1[:, t].reshape(L_N, L_N),
            interpolation = 'nearest', cmap = plt.cm.gray_r)
plt.title('Exponential Moving Average of Spikes')
#plt.colorbar()
plt.savefig('img.png', dpi = 400)
#plt.show()







#mean = np.mean(R, axis = 1)
#plt.imshow(mean.reshape(L_N, L_N), 
#           interpolation = 'nearest', cmap = plt.cm.gray_r)
#plt.show()
           