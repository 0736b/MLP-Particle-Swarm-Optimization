from mlp.mlp import MLP
import numpy as np
from copy import copy

class PSO:
    """Particle Swarm Optimization (Local Best)
    """
    
    def __init__(self, particle_size: int, max_t: int, layers_and_nodes: list, dataset: list):
        self.particle_size = particle_size
        self.max_t = max_t
        self.layers_and_nodes = layers_and_nodes
        self.dataset = dataset
        self.particles = []
        self.t = 0
        self.fs = []
        self.pbests = [float('inf')] * self.particle_size
        self.lbests = [float('inf')] * self.particle_size
        self.vs = [0.0] * self.particle_size
        self.weights_at_i = 0
        
    def init_particles(self):
        self.xpbests = []
        self.xlbests = []
        for i in range(self.particle_size):
            particle = MLP(self.layers_and_nodes)
            self.particles.append(particle)
            weights = particle.get_weights().copy()
            self.xpbests.append(weights)
            self.xlbests.append(weights)
            f = self.particles[i].run(self.dataset)
            self.fs.append(f)
    
    def calc_f(self, i):
        if not (self.t == 0):
            f = self.particles[i].run(self.dataset)
            self.fs.append(f)
        self.weights_at_i = self.particles[i].get_weights().copy()
        
    def compare_pbest_lbest(self, i):
        if self.fs[i] < self.pbests[i]:
            self.pbests[i] = self.fs[i]
            self.xpbests[i] = self.weights_at_i
        if self.fs[i] < min(self.fs[i-1], self.fs[(i+1) % self.particle_size]):
            self.lbests[i] = self.fs[i]
            self.xlbests[i] = self.weights_at_i
                
    def get_velocity(self, i):
        c1 = 1
        c2 = 2
        rho1 = np.random.uniform(0,1) * c1
        rho2 = np.random.uniform(0,1) * c2
        rho1_sub = np.subtract(self.xpbests[i], self.weights_at_i)
        rho2_sub = np.subtract(self.xlbests[i], self.weights_at_i)
        rho1_mul = rho1 * rho1_sub
        rho2_mul = rho2 * rho2_sub
        rho_sum = np.add(rho1_mul, rho2_mul)
        if self.t == 0:
            self.vs[i] = rho_sum
        else:
            self.vs[i] = np.add(self.vs[i], rho_sum)
                
    def tuning(self, i):
        new_weights = np.add(self.weights_at_i, self.vs[i]).tolist()
        self.particles[i].set_new_weights(new_weights)
            
    def run(self):
        self.init_particles()
        while self.t <= self.max_t:
            for i in range(0, self.particle_size, 1):
                self.calc_f(i)
                self.compare_pbest_lbest(i)
                self.get_velocity(i)
                self.tuning(i)
            print('@',self.t)
            self.t += 1
        best_particle = self.evaluate()
        return best_particle
            
    def evaluate(self):
        lowest_mae = float('inf')
        best_particle = None
        for particle in self.particles:
            mae = particle.run(self.dataset)
            if mae <= lowest_mae:
                lowest_mae = mae
                best_particle = copy(particle)
        print('Best Particle MAE:', lowest_mae)
        print('Best Particle Weights:', best_particle.get_weights())
        return best_particle