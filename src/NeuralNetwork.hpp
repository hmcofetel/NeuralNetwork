#ifndef NEURAL_NETWORK__
#define NEURAL_NETWORK__

#include "LayerDense.hpp"
#include "ActivationFunctions.hpp"
#include "MatrixXd.hpp"
#include <iostream>
#include <vector>

struct LayerDenseData
{
    LayerDense layer;
    MatrixXd delta;
    MatrixXd activation; //z1
    MatrixXd input;
    LayerDenseData(LayerDense layer) : layer(layer){};
};

class NeuralNetwork
{
private:
    std::vector<uint> __typology;
    std::vector<LayerDenseData> __layerDenseData;
    MatrixXd __input;
    MatrixXd __output;
    MatrixXd __buffer;
    Scalar __learningRate;
    Scalar __cost;

public:
    NeuralNetwork(std::vector<uint> typology, Scalar learningRate);
    void setData(MatrixXd input, MatrixXd output);
    void forward();
    void backward();
    Scalar cost();
};

NeuralNetwork::NeuralNetwork(std::vector<uint> typology, Scalar learningRate) : __typology(typology), __learningRate(learningRate)
{
    for (uint i = 0; i < __typology.size() - 1; i++)
    {
        __layerDenseData.push_back(LayerDenseData(LayerDense(typology[i], typology[i + 1])));
    }
}
void NeuralNetwork::setData(MatrixXd input, MatrixXd output)
{
    __input = input;
    __output = output;
    __buffer = input;
}

void NeuralNetwork::forward()
{

    for (std::vector<LayerDenseData>::iterator it = __layerDenseData.begin() ; it != __layerDenseData.end();++it)
    {
        
        (*it).input = __buffer;
        (*it).layer.forward(__buffer);
        (*it).activation = ActivationFunction::Sigmoid::forward((*it).layer.getOutput());
        __buffer = (*it).activation;
        

    }
    __cost = (__output - __buffer).array().mean();
    __buffer = __output - __buffer;
}

void NeuralNetwork::backward()
{
    for (std::vector<LayerDenseData>::reverse_iterator it = __layerDenseData.rbegin(); it != __layerDenseData.rend(); ++it)
    {
        (*it).delta = __buffer.array()*ActivationFunction::Sigmoid::backward((*it).activation).array();
        __buffer = (*it).delta*(*it).layer.getWeights().transpose();
        (*it).layer.setWeights((*it).layer.getWeights() + __learningRate*(*it).input.transpose()*(*it).delta);
    }
    __buffer = __input;
}

Scalar NeuralNetwork::cost()
{
    return __cost;
}

void train()
{
}
#endif // NEURAL_NETWORK__