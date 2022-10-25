from copy import copy
from mlp.mlp import MLP
import numpy as np

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
        self.xpbests = [None] * self.particle_size
        self.xlbests = [None] * self.particle_size
        self.vs = [0.0] * self.particle_size
        
    def init_particles(self):
        for i in range(self.particle_size):
            particle = MLP(self.layers_and_nodes)
            self.particles.append(particle)
    
    def calc_f(self):
        for particle in self.particles:
            f = particle.run(self.dataset)
            self.fs.append(f)
        
    def compare_pbest_lbest(self):
        for i in range(self.particle_size):
            if self.fs[i] < self.pbests[i]:
                self.pbests[i] = self.fs[i]
                self.xpbests[i] = self.particles[i].get_weights().copy()
            if self.fs[i] < np.min(self.fs[i-1], self.fs[i+1]):
                self.lbests[i] = self.fs[i]
                self.xlbests[i] = self.particles[i].get_weights().copy()
                
    def get_velocity(self):
        c1 = 1
        c2 = 2
        rho1 = np.random.uniform(0,1) * c1
        rho2 = np.random.uniform(0,1) * c2
        for i in range(self.particle_size):
            if self.t == 0:
                self.vs[i] = ((rho1 * np.add(np.subtract(self.xpbests[i], self.particles[i].get_weights()))), (rho2 * (np.subtract(self.xlbests[i], self.particles[i].get_weights()))))
            else:
                self.vs[i] = np.add(((rho1 * np.add(np.subtract(self.xpbests[i], self.particles[i].get_weights()))), (rho2 * (np.subtract(self.xlbests[i], self.particles[i].get_weights())))), self.vs[i])
                
    def tuning(self):
        for i in range(self.particle_size):
            new_weights = np.add(self.particles[i].get_weights(), self.vs[i])
            self.particles[i].set_new_weights(new_weights)
            
    def run(self):
        self.init_particles()
        while self.t <= self.max_t:
            self.calc_f()
            self.compare_pbest_lbest()
            self.get_velocity()
            self.tuning()
            self.t += 1
        self.evaluate()
            
    def evaluate(self):
        for particle in self.particles:
            mae = particle.run(self.dataset)
            print(mae)