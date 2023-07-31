#include "Dynamics.hpp"

#include <unistd.h>

Dynamics::Dynamics(int sim_time, int sim_steps, double damping, double stiffness)
    : simTime{sim_time}, simSteps{sim_steps}, dampingCoeff{damping}, stiffnessCoeff{stiffness} {}

Eigen::VectorXf Dynamics::getStateVector(Graph &g) const
{
    int nNodes = g.nodes.size();
    Eigen::VectorXf vec(nNodes);
    for (int i{0}; i < nNodes; i++)
    {
        vec(i) = g.nodes[i]->z;
    }
    return vec;
}

void Dynamics::setNodeStates(Graph &g, Eigen::VectorXf &states) const
{
    for (int i{0}; i < states.size(); i++)
    {
        g.nodes[i]->z_old = g.nodes[i]->z;
        g.nodes[i]->z = states(i);
    }
}

void Dynamics::runCentralizedDynamics(Graph &g, Force &force, Plot &plot) const
{
    plot.displayMethod("Centralized");
    int nNodes = g.nodes.size();
    Eigen::MatrixXf A_Matrix = dampingCoeff * (g.laplacianMatrix + epsilon * Eigen::MatrixXf::Identity(nNodes, nNodes));
    Eigen::MatrixXf B_Matrix = stiffnessCoeff * (g.laplacianMatrix + epsilon * Eigen::MatrixXf::Identity(nNodes, nNodes));
    Eigen::VectorXf x = getStateVector(g);
    Eigen::VectorXf x_dot = Eigen::VectorXf::Zero(nNodes);
    Eigen::VectorXf x_ddot(nNodes);
    double timeStep = double(simTime) / simSteps;
    for (int i{0}; i < simSteps + 1; i++)
    {
        x_ddot = force.sinusoidalForce(i * timeStep) - A_Matrix * x_dot - B_Matrix * x;
        // x_ddot = getForcing(i * timeStep, 1, 6.13602, nNodes) - A_Matrix * x_dot - B_Matrix * x;
        x += (x_dot * timeStep);
        x_dot += (x_ddot * timeStep);
        setNodeStates(g, x);
        for (int j{0}; j < nNodes; j++)
        {
            plot.plotNode(*g.nodes[j]);
            plot.displayState(*g.nodes[j]);
        }
        // std::cout << x << std::endl
        //           << std::endl;
        plot.displayTime(std::to_string(i * timeStep) + " s");
        plot.displayPlot();
        usleep(1E+3 * timeStep);
    }
}

void Dynamics::runDecentralizedDynamics(std::vector<std::shared_ptr<Node>> &nodes, Force &force, Plot &plot) const
{
    plot.displayMethod("Decentralized");
    double timeStep = double(simTime) / simSteps;
    for (int i{0}; i < simSteps + 1; i++)
    {
        Eigen::VectorXf force_vec = force.sinusoidalForce(i * timeStep);
        for (int j{0}; j < nodes.size(); j++)
        {
            double neighbor_z_sum{0}, neighbor_zdot_sum{0};
            for (int k{0}; k < nodes[j]->neighbors.size(); k++)
            {
                neighbor_z_sum += nodes[j]->neighbors[k]->z_old;
                neighbor_zdot_sum += nodes[j]->neighbors[k]->z_dot_old;
            }
            double z_ddot = force_vec(j) - dampingCoeff * (nodes[j]->neighbors.size() * nodes[j]->z_dot - neighbor_zdot_sum + epsilon * nodes[j]->z_dot) - stiffnessCoeff * (nodes[j]->neighbors.size() * nodes[j]->z - neighbor_z_sum + epsilon * nodes[j]->z);
            nodes[j]->z += (nodes[j]->z_dot * timeStep);
            nodes[j]->z_dot += (z_ddot * timeStep);
        }
        for (int j{0}; j < nodes.size(); j++)
        {
            nodes[j]->z_old = nodes[j]->z;
            nodes[j]->z_dot_old = nodes[j]->z_dot;
            plot.plotNode(*nodes[j]);
            plot.displayState(*nodes[j]);
        }

        plot.displayTime(std::to_string(i * timeStep) + " s");
        plot.displayPlot();
        usleep(1E+2 * timeStep);
    }
}