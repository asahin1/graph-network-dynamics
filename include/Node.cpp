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
    Eigen::MatrixXf gradientMat = Eigen::MatrixXf::Zero(localAdjMatrix.rows(),localAdjMatrix.cols());
    double sumEig = std::accumulate(eigenValues.begin(), eigenValues.end(), 0.0);
    int nEig = eigenValues.size();
    double a = 0.1;
    Eigen::VectorXf sumOverL = Eigen::VectorXf::Zero(nEig);
    for(int i{0}; i<nEig; i++){
        double lambda_i = eigenValues[i];
        for(int l{0}; l<nEig; l++){
            double lambda_l = eigenValues[l];
            sumOverL[i] += (-4*a*a *(a*a*lambda_i+lambda_i-lambda_l)) / (lambda_l * pow(a*a*lambda_i+pow(sqrt(lambda_i)-sqrt(lambda_l),2),3))+(4*a*a*lambda_l*(-a*a*lambda_l-lambda_l+4*sqrt(lambda_l)*sqrt(lambda_i)-3*lambda_i)) / (lambda_i * lambda_i * pow(a*a*lambda_l+pow(sqrt(lambda_l)-sqrt(lambda_i),2),3));
        }
    }
    for(int j{0}; j<gradientMat.rows(); j++){
        auto u_row_j = eigenVectors.row(j);
        for(int k{j+1}; k<gradientMat.cols(); k++){
            auto u_row_k = eigenVectors.row(k);
            if(localConMatrix(j,k) == 0)
                continue;
            double gradAtJK{0};
            for(int i{0}; i<nEig; i++){
                gradAtJK += sumOverL[i]*pow(u_row_j[i]-u_row_k[i],2);//gradJ*pow(u_row_j[i]-u_row_k[i],2) + gradK*pow(u_row_j[i]-u_row_k[i],2);
            }
            // std::cout << "Grad at (" << j << ", " << k << "): " << gradAtJK << std::endl;
            gradientMat(j,k) = gradAtJK;
            gradientMat(k,j) = gradAtJK;
            //}
    }
}
    // std::cout << "gradientMat:\n" << gradientMat << std::endl;
    double gradientNorm = gradientMat.norm();
    auto normalizedAndScaledGradient = gradientStep*2*gradientMat/gradientNorm;
    auto newAdjacencyMatrix = oldAdjacencyMatrix - normalizedAndScaledGradient;
    // std::cout << "newAdjacencyMatrix:\n" << newAdjacencyMatrix << std::endl;

    double matrixSum = oldAdjacencyMatrix.sum();
    Eigen::MatrixXf scaledAdjacencyMatrix = newAdjacencyMatrix*matrixSum/newAdjacencyMatrix.sum();
    std::vector<MatrixIdx> invalidWeights;
    for(int j{0}; j<scaledAdjacencyMatrix.rows(); j++){
        for(int k{j+1}; k<scaledAdjacencyMatrix.cols(); k++){
            if(localConMatrix(j,k) && scaledAdjacencyMatrix(j,k) < minEdgeWeight)
                invalidWeights.push_back(MatrixIdx(j,k));
        }
    }
    if(!invalidWeights.empty()){
        double minWeightVal = newAdjacencyMatrix.maxCoeff();
        for(int i{0}; i<newAdjacencyMatrix.rows(); i++){
            for(int j{0}; j<newAdjacencyMatrix.cols(); j++){
                if(localConMatrix(i,j)){
                    if(newAdjacencyMatrix(i,j)<minWeightVal)
                        minWeightVal = newAdjacencyMatrix(i,j);
                }
            }
        }
        double scaleFactor = (matrixSum - localConMatrix.sum()*minEdgeWeight)/(newAdjacencyMatrix.sum() - localConMatrix.sum()*minWeightVal);
        for(int i{0}; i<scaledAdjacencyMatrix.rows(); i++){
            for(int j{0}; j<scaledAdjacencyMatrix.cols(); j++){
                if(localConMatrix(i,j)){
                    scaledAdjacencyMatrix(i,j) = minEdgeWeight + scaleFactor*(newAdjacencyMatrix(i,j) - minWeightVal);
                }
            }
        }
    }
    // std::cout << "scaledAdjacencyMatrix:\n" << scaledAdjacencyMatrix << std::endl;
    invalidWeights.clear();
    for(int j{0}; j<scaledAdjacencyMatrix.rows(); j++){
        for(int k{j+1}; k<scaledAdjacencyMatrix.cols(); k++){
            if(localConMatrix(j,k) && scaledAdjacencyMatrix(j,k) < minEdgeWeight)
                invalidWeights.push_back(MatrixIdx(j,k));
        }
    }
    for(auto &el: invalidWeights)
        std::cout << "Invalid weight at (" << el.j << ", " << el.k << "): " << scaledAdjacencyMatrix(el.j,el.k) << std::endl;
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
