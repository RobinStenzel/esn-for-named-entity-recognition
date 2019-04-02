import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class ClassicReservoirTopology:
    def __init__(self, size):
        self.size = size

    def generateWeightMatrix(self, scaling=0.5):
        reservoirWeightRandom = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        return reservoirWeightRandom

class ClassicInputTopology:
    def __init__(self, inputSize, reservoirSize):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize

    def generateWeightMatrix(self, scaling = 0.5):
        inputWeightRandom = np.random.uniform(-scaling, +scaling, self.reservoirSize * self.inputSize).reshape((self.reservoirSize, self.inputSize))
        return inputWeightRandom

class RandomInputTopology:
    def __init__(self, inputSize, reservoirSize, inputConnectivity = 1.0):
        self.inputSize = inputSize
        self.reservoirSize = reservoirSize
        self.inputConnectivity = inputConnectivity
        self.connMatrix = None
        print("use seeding for consistent training results")
        np.random.seed(0)

    def generateConnectivityMatrix(self):
        connectivity = np.zeros((self.reservoirSize, self.inputSize))
        for i in range(self.reservoirSize):
            indices = np.random.choice(self.inputSize, size=int(np.ceil(self.inputConnectivity * self.inputSize)), replace=False)
            connectivity[i, indices] = 1.0
        return connectivity

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.reservoirSize * self.inputSize).reshape((self.reservoirSize, self.inputSize))
        self.connMatrix = self.generateConnectivityMatrix()
        weight = random * self.connMatrix
        return weight

class RandomReservoirTopology:
    def __init__(self, size, connectivity=1.0):
        self.size = size
        self.connectivity = connectivity
        self.connectivityMatrix = np.zeros((self.size, self.size))
        print("use seeding for consistent training results")
        np.random.seed(0)

        for i in range(self.size):
            indices = np.random.choice(self.size, size=int(np.ceil(self.connectivity * self.size)), replace=False)
            self.connectivityMatrix[i, indices] = 1.0

        # Create the network using connectivity matrix
        self.network = nx.from_numpy_matrix(np.matrix(self.connectivityMatrix))

        # Network stats object
        self.networkStats = Networkstats(self.network, self.size)

    def generateConnectivityMatrix(self):
        return self.connectivityMatrix

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        weight = random * self.connectivityMatrix
        return weight

class ErdosRenyiTopology:
    def __init__(self, size, probability):
        self.size = size
        self.probability = probability
        self.network = nx.erdos_renyi_graph(self.size, self.probability, directed=False)

        # Network stats object
        self.networkStats = Networkstats(self.network, self.size)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        weight = random * self.generateConnectivityMatrix()
        return weight

class SmallWorldGraphs:
    def __init__(self, size, meanDegree, beta):
        self.size = size
        self.meanDegree = meanDegree
        self.beta = beta
        self.network = nx.newman_watts_strogatz_graph(self.size,self.meanDegree,self.beta) #No edges are removed in newman implementation (So, atleast we get a ring lattice)

        # Network stats object
        self.networkStats = Networkstats(self.network, self.size)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        weight = random * self.generateConnectivityMatrix()
        return weight

class ScaleFreeNetworks:
    def __init__(self, size, attachmentCount):
        self.size = size
        self.m = attachmentCount
        self.network = nx.barabasi_albert_graph(self.size, self.m)

        # Network stats object
        self.networkStats = Networkstats(self.network, self.size)

    def generateConnectivityMatrix(self):
        connectivity = np.asarray(nx.to_numpy_matrix(self.network))
        return connectivity

    def generateWeightMatrix(self, scaling=0.5):
        # Multiply with randomly generated matrix with connected matrix
        random = np.random.uniform(-scaling, +scaling, self.size * self.size).reshape((self.size, self.size))
        weight = random * self.generateConnectivityMatrix()
        return weight

class Networkstats():

    def __init__(self, network, size):
        self.network = network
        self.size = size

    def getAverageDegree(self,):
        degrees = self.network.degree()
        sum_of_edges = sum(degrees.values())
        average_degree = sum_of_edges/ self.size

        # Normalize by the number of nodes in the network
        return average_degree/self.size

    def getAveragePathLenth(self):
        # Normalize by the number of nodes in the network
        return nx.average_shortest_path_length(self.network)/self.size

    def getAverageClusteringCoefficient(self):
        # Normalize by the number of nodes in the network
        return nx.average_clustering(self.network)/self.size

    def getDiameter(self):
        # Normalize by the number of nodes in the network
        return nx.diameter(self.network)/self.size







