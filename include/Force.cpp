#include "Force.hpp"

Force::Force(double amp, double freq, int size) : amplitude{amp}, frequency{freq}, size{size} {}

void Force::insertForceElement(int idx)
{
    nonZeroElements.push_back(idx);
}

Eigen::VectorXf Force::sinusoidalForce(double t)
{
    Eigen::VectorXf force(size);
    for (int i{0}; i < size; i++)
    {
        force(i) = 0;
    }
    for (int i{0}; i < nonZeroElements.size(); i++)
    {
        int idx = nonZeroElements[i];
        force(idx) = amplitude * sin(frequency * t);
    }
    return force;
}

Eigen::VectorXf Force::sinCauchyForce(double t)
{
    Eigen::VectorXf force(size);
    for (int i{0}; i < size; i++)
    {
        force(i) = 0;
    }
    std::vector<double> freq_vec;
    for(int i = 0; i < 100; i++){
        freq_vec.push_back(((double)rand() / (double)RAND_MAX));
    }
    double func = 0;
    for(int i = 0; i < 100; i++){
        //func = func + sin(2*M_PI*frequency*freq_vec[i]*t);
        func = func + sin(frequency*freq_vec[i]*t);
    }
    for (int i{0}; i < nonZeroElements.size(); i++)
    {
        int idx = nonZeroElements[i];
        force(idx) = amplitude*func;
    }
    return force;
}