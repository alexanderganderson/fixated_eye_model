import unittest
import particle_filter_new as PF
import numpy as np


class InitialProposalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.sigma = 0.1
        self.D_H = 2
        self.D_O = 4
        self.N_S = 50
        self.pd = PF.GaussIPD(self.D_H, self.D_O, self.sigma)
    
    def testVarInit(self):
        """
        Tests if variables initialized correctly
        """
        self.assertEquals(self.pd.D_H, self.D_H)
        self.assertEquals(self.pd.D_O, self.D_O)
        
        
    def testProb(self):
        X0 = np.zeros((self.N_S, self.D_H))
        Y0 = np.zeros((self.D_O,))
        probs = (np.ones(self.N_S) / 
                 np.sqrt(2 * np.pi * self.sigma ** 2) ** self.D_H)
        self.failUnlessAlmostEqual(self.pd.prob(X0, Y0)[0], 
                                   probs[0], places = 4)
        
    def testSample(self):
        Y0 = np.ones((self.D_O,))
        m = np.mean(self.pd.sample(Y0, self.N_S))
        self.failUnlessAlmostEqual(m, 0., places = 1)
        

class TransitionProposalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.D_H = 2
        self.D_O = self.D_H
        self.N_S = 100
        self.sigma = 0.1
        self.tpd = PF.GaussTPD(self.D_H, self.D_O, self.sigma)

    def testInit(self):
        self.assertEquals(self.D_H, self.tpd.D_H)
        self.assertEquals(self.D_O, self.tpd.D_O)
        self.assertEquals(self.sigma, self.tpd.sigma)

    def testProb(self):
        Xc = np.zeros((self.N_S, self.D_H))
        Yc = np.zeros((self.D_O,))
        Xp = np.zeros((self.N_S, self.D_H))
        expt = (np.ones((self.N_S,)) / 
                np.sqrt(2 * np.pi * self.sigma ** 2) ** self.D_H)
        self.failUnlessAlmostEqual(expt[0], self.tpd.prob(Xc, Yc, Xp)[0])

    def testSample(self):
        Xp = np.ones((self.N_S, self.D_H))
        Yc = np.zeros((self.D_O,))
        m = np.mean(self.tpd.sample(Yc, Xp), axis = 0)
        self.failUnlessAlmostEqual(Xp[0, 0], m[0], places = 1)
        

class InitialPotentialTest(unittest.TestCase):
    def setUp(self):
        self.D_H = 4
        self.N_S = 1
        self.sigma = 1.5
        self.ip = PF.GaussIP(self.D_H, self.sigma)

    def testInit(self):
        self.assertEquals(self.D_H, self.ip.D_H)

    def testProb(self):
        X0 = np.zeros((self.N_S, self.D_H))
        ans = 1. / np.sqrt(2 * np.pi * self.sigma ** 2) ** self.D_H
        self.failUnlessAlmostEqual(ans, self.ip.prob(X0)[0], places = 3)

class LikelihoodPotentialTest(unittest.TestCase):
    def setUp(self):
        self.D_H = 3
        self.D_O = self.D_H
        self.N_S = 10
        self.sigma = 1.5
        self.lp = PF.GaussLP(self.D_H, self.D_O, self.sigma)

    def testInit(self):
        self.assertEquals(self.D_O, self.lp.D_O)
        self.assertEquals(self.D_H, self.lp.D_H)
        self.assertEquals(self.sigma, self.lp.sigma)

    def testProb(self):
        Yc = np.zeros((self.D_O))
        Xc = np.zeros((self.N_S, self.D_H))
        ans = self.lp.prob(Yc, Xc)
        ans1 = 1. / np.sqrt(2 * np.pi * self.sigma ** 2) ** self.D_H
        self.failUnlessAlmostEqual(ans[0], ans1, places = 3)

class ParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.D_H = 3
        self.D_O = self.D_H
        self.sigma_trans = 0.1
        self.sigma_obs = 0.3
        self.N_T = 100
        self.N_P = 25

        self.ipd = PF.GaussIPD(self.D_H, self.D_O, self.sigma_trans)
        self.tpd = PF.GaussTPD(self.D_H, self.D_O, self.sigma_trans)
        self.ip = PF.GaussIP(self.D_H, self.sigma_trans)
        self.tp = PF.GaussTP(self.D_H, self.sigma_trans)
        self.lp = PF.GaussLP(self.D_H, self.D_O, self.sigma_obs)
        
        # Generate some data according to this model            
        self.X = np.zeros((self.N_T, self.D_H))
        self.Y = np.zeros((self.N_T, self.D_H))
        for i in range(self.D_H):
            self.X[:, i] = np.cumsum(self.sigma_trans * 
                                np.random.randn(self.N_T))
        self.Y = np.random.randn(self.N_T, self.D_H) * self.sigma_obs + self.X

        self.pf = PF.ParticleFilter(self.ipd, self.tpd, self.ip, 
                                    self.tp, self.lp, self.Y, self.N_P)

    def testInit(self):
        self.assertEquals(self.ipd, self.pf.ipd)
        self.assertEquals(self.tpd, self.pf.tpd)
        self.assertEquals(self.ip, self.pf.ip)
        self.assertEquals(self.tp, self.pf.tp)
        self.assertEquals(self.lp, self.pf.lp)

    def testReset(self):
        self.pf.reset()
        self.assertEquals(np.sum(self.pf.XS), 0)
        self.assertEquals(np.sum(self.pf.WS), 0)
        self.assertEquals(self.pf.t, 0)
        self.assertEquals(self.pf.resample_times, [])
        self.assertEquals(np.sum(self.pf.means), 0.)
        self.assertEquals(np.sum(self.pf.sdevs), 0.)

    def testResample_idx(self):
        W = np.ones((self.N_P,))
        self.assertRaises(ValueError, self.pf.resample_idx, W)

    def testAdvance(self):
        for _ in range(self.N_T):
            self.pf.advance()
        self.assertRaises(StopIteration, self.pf.advance)

    def testCalcMeanSdevs(self):
        for _ in range(self.N_T):
            self.pf.advance()
        self.assertNotEqual(np.mean(self.pf.WS ** 2), 0.)
        self.pf.calculate_means_sdevs()
        err = np.sqrt(np.mean((self.X - self.pf.means) ** 2))
        self.assertTrue(err < self.sigma_obs)

    def testPlot(self):
        self.pf.plot(self.X, 0.001, show = False)

    def testMain(self):
        PF.main()

if __name__ == '__main__':
    unittest.main()
