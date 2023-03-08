#ifndef ACTIVATION_FUNCTIONS__
#define ACTIVATION_FUNCTIONS__

#include "MatrixXd.hpp"

namespace ActivationFunction
{
    class ReLU
    {
    public:
        static MatrixXd forward(const MatrixXd &x)
        {
            return (x.array() < 0).select(0, x);
        }

        static MatrixXd backward(const MatrixXd &x)
        {
            return (x.array() < 0).select(0, MatrixXd::Ones(x.rows(), x.cols()));
        }
    };

    class SoftMax
    {
    public:
        static MatrixXd forward(const MatrixXd &x)
        {
            MatrixXd expX = x.array().exp();
            MatrixXd y = x;
            for (int row = 0; row < x.rows(); ++row)
            {
                y.row(row) = expX.row(row) / expX.row(row).sum();
            }
            return y;
        }
    };

    class Sigmoid
    {
    public:
        static MatrixXd forward(const MatrixXd &x)
        {
            return 1.0 / (1.0 + (-x.array()).exp());
        }

        static MatrixXd backward(const MatrixXd &x)
        {
            return  forward(x).array() * (1 - forward(x).array());
        }
    };
}

#endif // ACTIVATION_FUNCTIONS__