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