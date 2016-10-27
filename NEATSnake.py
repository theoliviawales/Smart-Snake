from neat import nn, population, statistics
import os

def eval_fitness(genomes):
    net = nn.create_feed_forward_phenotype(g)
    sum_square_error = 0.0
    for inputs, expected in zip(xor_inputs, xor_outputs):
        # Serial activation propagates the inputs through the entire network.
        output = net.serial_activate(inputs)
        sum_square_error += (output[0] - expected) ** 2

    # When the output matches expected for all inputs, fitness will reach
    # its maximum value of 1.0.
    g.fitness = 1 - sum_square_error
