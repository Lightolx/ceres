#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>

using std::cout;
using std::endl;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;

// 求解powell函数
// f1(x) = x1 + 10*x2
// f2(x) = √5 * (x3 - x4)
// f3(x) = (x2 - 2*x3)^2
// f4(x) = √10 * (x1 - x4)^2
// F(x) = [f1(x), f2(x), f3(x), f4(x)  F(x)其实是一个四维向量，求让这个向量的模最小的[x1, x2, x3, x4]的值


struct CostFunctor
{
    template <typename T>
    bool operator()(const T* const x, T* residual) const
    {
        residual[0] = x[0] - T(10)*x[1];
        residual[1] = T(std::sqrt(5)) * (x[2] - x[3]);
        // 不要对x调用其他库的函数，因为对x要算Jacobian，其他的库在写这个
        // 函数时肯定没有重载过输入是ceres::Jet的情况
//        residual[2] = std::pow((x[1] - T(2)*x[2]), 2);
        residual[2] = (x[1] - T(2)*x[2]) * (x[1] - T(2)*x[2]);
        // 对于常量可以调用其他库的函数，因为就算是计算Jacobian时，常量也能自动化为单位阵
        residual[3] = T(std::sqrt(10)) * (x[0] - x[3]) * (x[0] - x[3]);

        return true;
    }
};

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    double x1 = 3.0;
    double x2 = -1.0;
    double x3 = 0.0;
    double x4 = 1.0;

    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 4, 4>(new CostFunctor);
    Problem problem;
    problem.AddResidualBlock(cost_function, nullptr, &x1, &x2, &x3, &x4);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    cout << "x1: " << x1 << " -> " << x1 << endl;
    cout << "x2: " << x2 << endl;
    cout << "x3: " << x3 << endl;
    cout << "x4: " << x4 << endl;

    return 0;
}