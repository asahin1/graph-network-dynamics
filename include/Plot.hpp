#ifndef _PLOT_H
#define _PLOT_H

#include "Graph.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


class Plot
{
private:
    std::string windowName;
    cv::Mat blankImg;
    cv::Mat currentImg;

    int plotScale;
    int verticalPadding, horizontalPadding;

    int maxX, maxY;

    static CvScalar defaultNodeColor;
    static CvScalar defaultEdgeColor;

    static int defaultNodeSize;
    static double defaultEdgeThickness;

    cv::Point transformGraphToPlot(const Node &n) const;

public:
    Plot(std::string name, int scale, int vPadding, int hPadding, int xMax, int yMax);

    void displayTime(const std::string &time_str);

    void displayMethod(const std::string &method_str);

    void displayState(const Node &n);

    void displayEigenvectorVal(const Node &n, float val);

    void plotEigen(const Graph &g, Eigen::VectorXf &vec, int size = defaultNodeSize);

    void plotNode(Node &n, int size = defaultNodeSize);

    void plotEdge(Node &n1, Node &n2, double thickness, CvScalar color);

    void plotGraph(Graph &g);

    void initWindow();

    void displayPlot(bool waitForKey = false);

    double round(double var, int place);
};

#endif