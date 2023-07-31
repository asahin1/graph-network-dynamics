#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include "gnuplot-iostream.h"
#include <Eigen/Dense>

namespace histogram{
    void generateHistogram(Gnuplot &plotObj, Eigen::VectorXf &eigenvalues);
}

#endif
