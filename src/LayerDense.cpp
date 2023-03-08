#include "LayerDense.hpp"

LayerDense::LayerDense(uint n_inputs, uint n_neurons) : __n_inputs(n_inputs), __n_neurons(n_neurons)
{
    srand((uint) time(0));
    __weights = MatrixXd::Random(__n_inputs, __n_neurons);
    __bias = MatrixXd::Zero(1, __n_neurons);
}

void LayerDense::forward(MatrixXd inputs)
{
    __output = (inputs * __weights) + __bias.replicate(inputs.rows(), 1);
}

MatrixXd LayerDense::getOutput()
{
    return __output;
}

MatrixXd LayerDense::getWeights()
{
    return __weights;
}

MatrixXd LayerDense::getBias()
{
    return __bias;
}


void LayerDense::setOutput(MatrixXd output)
{
    __output = output;
}

void LayerDense::setWeights(MatrixXd weights)
{
    __weights = weights;
}

void LayerDense::setBias(MatrixXd bias)
{
    __bias = bias;
}