#include "LayerDense.hpp"
#include "ActivationFunctions.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>
int main(int argc, char *argv[] ){

    MatrixXd X(4,4);
    X << 12.0,9.0,1,2,
        1.0,5.0,10,50,
        3.0,6.0, 20,60,
        50,20,2,3
        ;


    MatrixXd Y(4,1);
    Y << 1,
        0,
        0,
        1;


    // LayerDense layer1(2, 3), layer2(3,1);

    
    // layer1.forward(X);
    // MatrixXd z1 = ActivationFunction::ReLU::forward(layer1.getOutput()); //

    // layer2.forward(z1);
    // MatrixXd ouput = ActivationFunction::ReLU::forward(layer2.getOutput());

    // // error in layer 2
    // MatrixXd ouput_error = Y - ouput;
    // std::cout << "ouput_error: \n"<<ouput_error << std::endl;

    // MatrixXd ouput_delta = (ouput_error.array() * ActivationFunction::ReLU::backward(ouput).array());
    // std::cout << "ouput_delta: \n"<<ouput_delta << std::endl;

    // MatrixXd layer2_error = ouput_delta*layer2.getWeights().transpose();
    // std::cout << "layer2_error: \n"<<layer2_error << std::endl;


    // // error in layer 1
    // MatrixXd layer2_delta = layer2_error.array() * ActivationFunction::ReLU::backward(z1).array();
    // std::cout << "layer2_delta: \n"<<layer2_delta << std::endl;
    
    // layer1.setWeights(layer1.getWeights() + X.transpose()*layer2_delta);
    // layer2.setWeights(layer2.getWeights() + z1.transpose()*ouput_delta);

    

    NeuralNetwork NN({4,8,1},0.05);
    NN.setData(X,Y);

    for (int i =0 ; i < std::stoi(argv[1]); i ++){
        NN.forward();
        std::cout << "cost: "<<NN.cost() << std::endl;
        NN.backward();
        
    }


}