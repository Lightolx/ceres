#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>

using std::cout;
using std::endl;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;

// 求解函数f(x) = (x - 10)^2 的最小值

struct CostFunctor
{
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        residual[0] = x[0] - T(10);

        return true;
    }
};

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    double x0 = 0.5;
    double x = x0;

    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    Problem problem;
    problem.AddResidualBlock(cost_function, nullptr, &x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    cout << "x: " << x0 << " -> " << x << endl;

    return 0;
}