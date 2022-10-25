from utils.datareader import get_dataset, cross_valid
from pso.pso import PSO

def main():
    dataset = get_dataset()
    pso = PSO(10,100,[])

if __name__ == '__main__':
    main()