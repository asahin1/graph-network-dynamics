#include "GradientDescent.hpp"
#include "Histogram.hpp"
#include "Plot.hpp"
#include <memory>
#include <numeric>
#include <vector>

GradientDescent::GradientDescent(std::shared_ptr<Graph> initGraph): graphGridSize{initGraph->gridSize}, constrainedWeights{true}, fixedWeightSum{true}{
    graphHistory.push_back(initGraph);
}

GradientDescent::GradientDescent(std::shared_ptr<Graph> initGraph, bool weightConstraint, bool weightSumConstraint): graphGridSize{initGraph->gridSize}, constrainedWeights{weightConstraint}, fixedWeightSum{weightSumConstraint}{
    graphHistory.push_back(initGraph);
}

void GradientDescent::runNStepDescent(const int nIter){
    
    plotHistogram();
    plotGraph();
    for(int i{0}; i<nIter; i++){
        printIterInfo(i);
        runOneStepDescent();
        if(graphHistory.size() < i+1){
            std::cout << "Stuck at iter: " << i+1 << std::endl;
            break;
        }
        plotHistogram();
    }
    plotGraph();
}

void GradientDescent::runOneStepDescent(){
    Eigen::MatrixXf oldAdjacencyMatrix = graphHistory.back()->adjacencyMatrix; 
    Eigen::MatrixXf newAdjacencyMatrix = oldAdjacencyMatrix; 
    Eigen::MatrixXf scaledAdjacencyMatrix = newAdjacencyMatrix; 
    std::vector<MatrixIdx> weightsToAvoid; 
    int nRecompute{0};
    do{
        if(nRecompute > maxRecompute)
            return;
        auto adjGradient = computeAdjGradientDoubleSumNew(graphHistory.back(), weightsToAvoid); 
        double sum_s = 0;
        for(int i{0}; i < adjGradient.col(0).size(); i++){
            sum_s = sum_s + adjGradient.col(i).dot(adjGradient.col(i));
        }
        sum_s = 1;
        newAdjacencyMatrix = oldAdjacencyMatrix - ((gradientStep/sum_s) * adjGradient); 
        if(fixedWeightSum)
            scaledAdjacencyMatrix = newAdjacencyMatrix * oldAdjacencyMatrix.sum() / newAdjacencyMatrix.sum(); 
        else
            scaledAdjacencyMatrix = newAdjacencyMatrix; 
        nRecompute++;
        if(constrainedWeights)
            weightsToAvoid = getInvalidWeightIdx(scaledAdjacencyMatrix);
    }while(!weightsToAvoid.empty());
    graphHistory.push_back(graphHistory.back()->applyGradient(scaledAdjacencyMatrix));
}

Eigen::MatrixXf GradientDescent::computeAdjGradientDoubleMin(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const{

    // double max gradient
    Eigen::MatrixXf gradientMat = Eigen::MatrixXf::Zero(graph->adjacencyMatrix.rows(),graph->adjacencyMatrix.cols());
    auto eigenvalues = graph->eigenValues;
    double minVal = eigenvalues[eigenvalues.size()-1];
    MatrixIdx closestLambdaIdx(-1,-1);
    for(int j{0}; j<eigenvalues.size(); j++){
        double lambda_j = graph->eigenValues[j];
        for(int k{j+1}; k<eigenvalues.size(); k++){
            double lambda_k = eigenvalues[k];
            if(lambda_k - lambda_j < minVal){
                minVal = lambda_k-lambda_j;
                closestLambdaIdx = MatrixIdx(j,k);
            }

        }
    }
    auto u_j = graph->eigenVectors.col(closestLambdaIdx.j);
    auto u_k = graph->eigenVectors.col(closestLambdaIdx.k);
    double lambda_j = eigenvalues[closestLambdaIdx.j];
    double lambda_k = eigenvalues[closestLambdaIdx.k];
    for(int i{0}; i<gradientMat.rows(); i++){
        for(int l{i+1}; l<gradientMat.cols(); l++){
            if(graph->connectivityMatrix(i,l) == 0)
                continue;
            gradientMat(i,l) = 2*(lambda_k-lambda_j+0.1)*(pow(u_k[i]-u_k[l],2) - pow(u_j[i]-u_j[l],2));
            gradientMat(l,i) = gradientMat(i,l);
        }
    }
    for(auto &el: weightsToAvoid){
        gradientMat(el.j,el.k) = 0;
        gradientMat(el.k,el.j) = 0;
    }
    return gradientMat;
}

Eigen::MatrixXf GradientDescent::computeAdjGradientDoubleSum(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const{

    // double sum gradient
    Eigen::MatrixXf gradientMat = Eigen::MatrixXf::Zero(graph->adjacencyMatrix.rows(),graph->adjacencyMatrix.cols());
    double sumEig = std::accumulate(graph->eigenValues.begin(), graph->eigenValues.end(), 0.0);
    int nEig = graph->eigenValues.size();
    for(int j{0}; j<gradientMat.rows(); j++){
        auto u_row_j = graph->eigenVectors.row(j);
        for(int k{j+1}; k<gradientMat.cols(); k++){
            auto u_row_k = graph->eigenVectors.row(k);
            if(graph->connectivityMatrix(j,k) == 0)
                continue;
            double gradAtJK{0};
            for(int i{0}; i<nEig; i++){
                double lambda_i = graph->eigenValues[i];
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
    return gradientMat;
}

Eigen::MatrixXf GradientDescent::computeAdjGradientDoubleSumNew(const std::shared_ptr<Graph> graph, std::vector<MatrixIdx> &weightsToAvoid) const{

    // double sum gradient
    Eigen::MatrixXf gradientMat = Eigen::MatrixXf::Zero(graph->adjacencyMatrix.rows(),graph->adjacencyMatrix.cols());
    double sumEig = std::accumulate(graph->eigenValues.begin(), graph->eigenValues.end(), 0);
    int nEig = graph->eigenValues.size();
    double a = 0.1;
    for(int j{0}; j<gradientMat.rows(); j++){
        auto u_row_j = graph->eigenVectors.row(j);
        for(int k{j+1}; k<gradientMat.cols(); k++){
           // if(j < k){
                auto u_row_k = graph->eigenVectors.row(k);
                if(graph->connectivityMatrix(j,k) == 0)
                    continue;
                double gradAtJK{0};
                double grad1{0};
                double grad2{0};
                for(int i{0}; i<nEig; i++){
                    double lambda_i = graph->eigenValues[i];
                    for(int l{0}; l<nEig; l++){
                        double lambda_l = graph->eigenValues[l];
                        if(i != l){
                           // grad1 += (-4*a*a *(a*a*lambda_i+lambda_i+lambda_l)) / (lambda_l * pow(a*a*lambda_i+pow(sqrt(lambda_i)-sqrt(lambda_l),2),3));
                           // grad2 += (4*a*a*lambda_l*(-a*a*lambda_l-lambda_l+4*sqrt(lambda_l)*sqrt(lambda_i)-3*lambda_i)) / (lambda_i * lambda_i * pow(a*a*lambda_l+pow(sqrt(lambda_l)-sqrt(lambda_i),2),3));
                            grad1 += (-4*a*a *(a*a*lambda_i+lambda_i-lambda_l)) / (lambda_l * pow(a*a*lambda_i+pow(sqrt(lambda_i)-sqrt(lambda_l),2),3));
                            grad2 += (4*a*a*lambda_l*(-a*a*lambda_l-lambda_l+4*sqrt(lambda_l)*sqrt(lambda_i)-3*lambda_i)) / (lambda_i * lambda_i * pow(a*a*lambda_l+pow(sqrt(lambda_l)-sqrt(lambda_i),2),3));
                        }
                    }
                    gradAtJK += (grad1+grad2)*pow(u_row_j[i]-u_row_k[i],2);//gradJ*pow(u_row_j[i]-u_row_k[i],2) + gradK*pow(u_row_j[i]-u_row_k[i],2);
                }
                gradientMat(j,k) = gradAtJK;
                gradientMat(k,j) = gradAtJK;
            //}
        }
    }
    for(auto &el: weightsToAvoid){
        gradientMat(el.j,el.k) = 0;
        gradientMat(el.k,el.j) = 0;
    }
    return gradientMat;
}

std::vector<MatrixIdx> GradientDescent::getInvalidWeightIdx(const Eigen::MatrixXf &adjMat) const{
    std::vector<MatrixIdx> res;
    for(int j{0}; j<adjMat.rows(); j++){
        for(int k{j+1}; k<adjMat.cols(); k++){
            if(graphHistory.back()->connectivityMatrix(j,k) && adjMat(j,k) < minEdgeWeight)
                res.push_back(MatrixIdx(j,k));
        }
    }
    return res; 
}

std::vector<double> GradientDescent::returnEigenvalues(){
    std::vector<double> ev;
    for(int i{}; i < graphGridSize*graphGridSize; i++){
        ev.push_back(graphHistory.back()->eigenValues[i]);
    }
    return ev;
}

void GradientDescent::plotHistogram(){

    if(!graphHistory.empty())
        histogram::generateHistogram(histogramStream,graphHistory.back()->eigenValues);

}

void GradientDescent::destroyHistogram(){
    histogramStream << "set term x11 close\n";
}

void GradientDescent::plotGraph(){

    Plot graphPlotter("Graph Plot", 500/graphGridSize, 2, 2, graphGridSize, graphGridSize);
    graphPlotter.plotGraph(*graphHistory.back());
    graphPlotter.displayPlot(true);

}

void GradientDescent::printIterInfo(const int iterNo) const{
    std::cout << "Iter #: " << iterNo+1 << std::endl;
    auto eigenvalues = graphHistory.back()->eigenValues;
    std::cout << "Eigenvalue sum: " << std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0) << std::endl;
    double eigDistNorm{0};
    double eigDistMin{eigenvalues[eigenvalues.size()-1]};
    for(int j{0}; j<eigenvalues.size(); j++){
        for(int k{0}; k<eigenvalues.size(); k++){
            eigDistNorm += pow(eigenvalues[j] - eigenvalues[k], 2);
            if(abs(eigenvalues[j]-eigenvalues[k])<eigDistMin)
                eigDistMin = abs(eigenvalues[j]-eigenvalues[k]);
        }
    }
    std::cout << "Eigenvalues cumulative distance : " << eigDistNorm << std::endl;
    std::cout << "Eigenvalues minimum distance : " << eigDistMin << std::endl;
}
