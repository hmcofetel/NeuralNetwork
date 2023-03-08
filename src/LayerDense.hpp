#ifndef LAYER_DENSE__
#define LAYER_DENSE__

#include "MatrixXd.hpp"

class LayerDense
{
private:
    uint __n_inputs;
    uint __n_neurons;
    MatrixXd __weights;
    MatrixXd __bias;
    MatrixXd __output;

public:
    LayerDense(uint n_inputs, uint n_neurons);
    void forward(MatrixXd inputs);

    MatrixXd getOutput();
    MatrixXd getWeights();
    MatrixXd getBias();

    void setWeights(MatrixXd weights);
    void setOutput(MatrixXd output);
    void setBias(MatrixXd bias);    

};

#endif