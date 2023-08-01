#include "Node.hpp"

#include <Eigen/Dense>
#include <math.h>
#include <iostream>
#include <memory>
#include <numeric>

Node::Node(int xx, int yy, int id) : x(xx), y(yy), id(id), z(0), z_old(0) {}
Node::Node(int xx, int yy, int id, double zz) : x(xx), y(yy), id(id), z(zz), z_old(zz) {}

double Node::getDist(const Node &n) const
{
    double dx = x - n.x;
    double dy = y - n.y;
    return sqrt(dx * dx + dy * dy);
}
double Node::isNear(const Node &n) const
{
    return getDist(n) < sqrt(2);
}

bool Node::isNeighbor(const std::shared_ptr<Node> n) const
{
    //     for (int i{0}; i < neighbors.size(); i++)
    //     {
    //         if (neighbors[i] == n)
    //             return true;
    //     }
    if(neighbors.find(n) != neighbors.end())
        return true;
    return false;
}

void Node::runLocalGD(){
    // compute and update own weight info
    Eigen::MatrixXf localAdjMatrix = Eigen::MatrixXf::Zero(neighbors.size()+1, neighbors.size()+1);
    Eigen::MatrixXf localConMatrix = Eigen::MatrixXf::Zero(neighbors.size()+1, neighbors.size()+1);
    int idx{1};
    for(auto &n: neighbors){
        localAdjMatrix(0,idx) = (*n.second);
        localAdjMatrix(idx,0) = localAdjMatrix(0,idx);
        localConMatrix(0,idx) = 1;
        localConMatrix(idx,0) = 1;
        idx++;
    }
    Eigen::MatrixXf localDegMatrix = Eigen::MatrixXf::Zero(neighbors.size()+1, neighbors.size()+1);
    for(int i{0}; i<localAdjMatrix.rows(); i++){
        localDegMatrix(i,i) = std::accumulate(localAdjMatrix.row(i).begin(), localAdjMatrix.row(i).end(), 0.0);
    }
    Eigen::MatrixXf localLapMatrix = localDegMatrix - localAdjMatrix;
    Eigen::MatrixXf matrixToSolve = localLapMatrix + 0.1 * Eigen::MatrixXf::Identity(localLapMatrix.rows(), localLapMatrix.cols());
    Eigen::EigenSolver<Eigen::MatrixXf> solver(matrixToSolve);
    Eigen::VectorXf eigenValues = solver.eigenvalues().real();
    Eigen::MatrixXf eigenVectors = solver.eigenvectors().real();

    // Combine eigenvalues and eigenvectors into a std::vector of pairs
    std::vector<std::pair<double, Eigen::VectorXf>> eigenPairs;
    for (int i = 0; i < eigenValues.size(); ++i) {
        eigenPairs.push_back(std::make_pair(eigenValues[i], eigenVectors.col(i)));
    }

    // Sort the vector of pairs based on eigenvalues in ascending order
    std::sort(eigenPairs.begin(), eigenPairs.end(), [](const std::pair<double, Eigen::VectorXf>& a,
                const std::pair<double, Eigen::VectorXf>& b) {
            return a.first < b.first;
            });

    // Extract the sorted eigenvalues and eigenvectors
    for (int i = 0; i < eigenValues.size(); ++i) {
        eigenValues[i] = eigenPairs[i].first;
        eigenVectors.col(i) = eigenPairs[i].second;
    }
    Eigen::MatrixXf oldAdjacencyMatrix = localAdjMatrix; 
    Eigen::MatrixXf newAdjacencyMatrix = oldAdjacencyMatrix; 
    Eigen::MatrixXf scaledAdjacencyMatrix = newAdjacencyMatrix; 
    std::vector<MatrixIdx> weightsToAvoid; 
    double gradientStep{0.1};
    int nRecompute{0};
    do{
        if(nRecompute > 5)
            return;
        Eigen::MatrixXf gradientMat = Eigen::MatrixXf::Zero(localAdjMatrix.rows(),localAdjMatrix.cols());
        double sumEig = std::accumulate(eigenValues.begin(), eigenValues.end(), 0.0);
        int nEig = eigenValues.size();
        for(int j{0}; j<gradientMat.rows(); j++){
            auto u_row_j = eigenVectors.row(j);
            for(int k{j+1}; k<gradientMat.cols(); k++){
                auto u_row_k = eigenVectors.row(k);
                if(localConMatrix(j,k) == 0)
                    continue;
                double gradAtJK{0};
                for(int i{0}; i<nEig; i++){
                    double lambda_i = eigenValues[i];
                    gradAtJK += 4*pow(u_row_j[i]-u_row_k[i],2)*(nEig*lambda_i - sumEig); 
                }
                gradientMat(j,k) = gradAtJK;
                gradientMat(k,j) = gradAtJK;
            }
        }
        for(auto &el: weightsToAvoid){
            gradientMat(el.j,el.k) = 0;
            gradientMat(el.k,el.j) = 0;
        }
        newAdjacencyMatrix = oldAdjacencyMatrix + gradientStep * gradientMat; 
        scaledAdjacencyMatrix = newAdjacencyMatrix * oldAdjacencyMatrix.sum() / newAdjacencyMatrix.sum(); 
        nRecompute++;
        weightsToAvoid.clear();
        for(int j{0}; j<scaledAdjacencyMatrix.rows(); j++){
            for(int k{j+1}; k<scaledAdjacencyMatrix.cols(); k++){
                if(localConMatrix(j,k) && scaledAdjacencyMatrix(j,k) < 0.2)
                    weightsToAvoid.push_back(MatrixIdx(j,k));
            }
        }
    }while(!weightsToAvoid.empty());
    idx = 1;
    for(auto &n: neighbors){
        *n.second = scaledAdjacencyMatrix(0,idx);
        idx++;
    }
}

void Node::print(const std::string head) const
{
    std::cout << head << " : "
              << "x = " << x << ", y = " << y << std::endl;
}
