import multiprocessing
from vec_mlp.vec_mlp import VEC_MLP
import numpy as np
from copy import copy
import warnings
warnings.filterwarnings("ignore")

num_workers = multiprocessing.cpu_count()

class PSO:
    """Particle Swarm Optimization (Global Best)
    """
    
    def __init__(self, particle_size: int, max_t: int, layers_and_nodes: list, dataset: list):
        """setups pso

        Args:
            particle_size (int): number of particles
            max_t (int): max epoch
            layers_and_nodes (list): layers and node for that particle
            dataset (list): dataset that use for train
        """
        self.log_result_gbest = []
        self.log_result_pbest_avg = []
        self.particle_size = particle_size
        self.usePool = self.particle_size >= 150
        self.max_t = max_t
        self.layers_and_nodes = layers_and_nodes
        self.dataset = dataset
        self.particles = []
        self.t = 1
        self.fs = [0] * self.particle_size
        self.vs = [0.0] * self.particle_size
        self.pbests = [float('inf')] * self.particle_size
        self.xpbests = [None] * self.particle_size
        self.gbests = float('inf')
        self.xgbests = None
        # self.c1 = np.random.uniform(0,1)
        # self.c2 = np.random.uniform(0,3)
        self.c1 = 1.0
        self.c2 = 2.0
        self.delta_t = 0.5
        self.all_i = [int(i) for i in range(self.particle_size)]
        print('[+] use pool:', self.usePool)
        
    def init_particles(self):
        """initialize all particle
        """
        for i in range(self.particle_size):
            particle = VEC_MLP(self.layers_and_nodes)
            self.particles.append(particle)
    
    def task_calc_f(self,mlp):
        """calculate fitness

        Args:
            mlp: particle

        Returns:
            f: fitness value
        """
        f = mlp.run(self.dataset[0], self.dataset[1])
        return f
        
    def calc_f(self):
        """calculate fitness of all particles
        """
        if self.usePool:
            result = []
            with multiprocessing.Pool(num_workers) as pool:
                result = pool.map(self.task_calc_f, self.particles)
            self.fs = result
            pool.close()
        else:
            for i in range(self.particle_size):
                f = self.particles[i].run(self.dataset[0], self.dataset[1])
                self.fs[i] = f
        
    def compare_pbest_gbest(self):
        """comparing pbest and gbest
        """
        for i in range(self.particle_size):
            if self.fs[i] < self.pbests[i]:
                self.pbests[i] = self.fs[i]
                self.xpbests[i] = self.particles[i].get_weights()
            if self.fs[i] < self.gbests:
                self.gbests = self.fs[i]
                self.xgbests = self.particles[i].get_weights()
                                
    def get_velocity(self):
        """calculate velocity
        """
        for i in range(self.particle_size):
            rho1 = np.random.uniform(0,1) * self.c1
            rho2 = np.random.uniform(0,1) * self.c2
            k = 1.0
            if (rho1+rho2) > 4.0:
                big_rho = rho1+rho2
                t_big_rho = big_rho
                k = 1 - (1/big_rho) + ((np.sqrt(np.abs(np.power(t_big_rho,2) - 4 * big_rho))) / 2)
            rho1_sub = np.subtract(self.xpbests[i], self.particles[i].get_weights())
            rho2_sub = np.subtract(self.xgbests, self.particles[i].get_weights())
            rho1_mul = rho1 * rho1_sub
            rho2_mul = rho2 * rho2_sub
            rho_sum = np.add(rho1_mul, rho2_mul)
            if self.t == 0:
                self.vs[i] = k * rho_sum
            else:
                self.vs[i] = k * (np.add((self.vs[i] * self.delta_t), rho_sum))
                
    def tuning(self):
        """calculate new weight with velocity of that particle
        """
        for i in range(self.particle_size):
            new_weights = (np.add(self.particles[i].get_weights(), self.vs[i]))
            self.particles[i].set_weights(new_weights)
            
    def run(self):
        """running pso training
        """
        self.init_particles()
        while self.t <= self.max_t:
            self.calc_f()
            self.compare_pbest_gbest()
            self.get_velocity()
            self.tuning()
            print('@',self.t, 'G-Best:', self.gbests, 'AVG P-Best:', np.average(self.pbests))
            self.log_result_gbest.append(copy(self.gbests))
            self.log_result_pbest_avg.append(copy(np.average(self.pbests)))
            self.t += 1
        best_particle = copy(self.xgbests)
        best_mae = self.gbests.copy()
        return best_particle, best_mae, self.log_result_gbest, self.log_result_pbest_avg