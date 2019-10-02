from src.perceptron import Perceptron, NearestNeighbor

def main():
    perceptron = Perceptron("mnist-x.data", "mnist-y.data")
    perceptron.train('7', '5', 100)
    print(perceptron.test('7', '5'))

    # nearest_neighbor = NearestNeighbor("mnist-x.data", "mnist-y.data")
    # print(nearest_neighbor.test())

if __name__ == '__main__':
    main()
