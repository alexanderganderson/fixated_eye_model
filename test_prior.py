import EM_burak_class as EBC

# Run EM algorithm for different values of
#    sparse prior strength

ALPHAs = [0., 0.1, 1., 10., 100., 1000.]
DC = 100.
DT = 0.001
N_T = 100

N_trials = 10

for ALPHA in ALPHAs:
    emb = EBC.EMBurak(DT = DT, DC = DC, ALPHA = ALPHA, N_T = N_T)
    for _ in range(N_trials):
        emb.gen_data()
        emb.run()
