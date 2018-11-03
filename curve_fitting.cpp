#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Eigen>
#include <ceres/ceres.h>
#include <glog/logging.h>

using std::cout;
using std::endl;
using ceres::AutoDiffCostFunction;

//  拟合曲线 y = exp(m*x + c)  x=1:0.01:0.5, m = 3, c = 0.75
// defining a templated object to evaluate the residual

struct costFunctor
{
    costFunctor(double x, double y): x_(x), y_(y)
    {}

    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const
    {
        residual[0] = T(y_) - ceres::exp(m[0]*x_ + c[0]);

        return true;
    }

    const double x_;
    const double y_;
};

int main()
{
    // step0: Read in data
    std::ifstream fin("y.txt");
    std::string ptline;
    double y;
    std::vector<double> X;
    std::vector<double> Y;

    double x = 0;
    while (getline(fin, ptline))
    {
        std::stringstream ss(ptline);
        ss >> y;
        Y.push_back(y);
        X.push_back(x);
        x += 0.01;
    }
    
    // Step1: construct the problem
    double m = 2; double m0 = m;
    double c = 1; double c0 = c;
    ceres::Problem problem;
    for (int i = 0; i < X.size(); ++i)
    {
        // 三个1, 1, 1分别表示residual的维度， 第一个优化变量c的维度， 第二个优化变量m的维度
        ceres::CostFunction* pCostFunction = new AutoDiffCostFunction<costFunctor, 1, 1, 1>(
                new costFunctor(X[i], Y[i]));
        
        problem.AddResidualBlock(pCostFunction, nullptr, &m, &c);
    }
    
    // Step2: configure options and solve the optimization problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    cout << "m: " << m0 << " -> " << m << endl;
    cout << "c: " << c0 << " -> " << c << endl;

    // Step3: 验证ground truth 与 最小二乘法的解谁的cost function要更小
    double mgt = 3;
    double cgt = 0.75;
    std::vector<double> errors1;
    std::vector<double> errors2;
    for (int i = 0; i < X.size(); ++i)
    {
        errors1.push_back(pow(Y[i] - ceres::exp(m*X[i] + c), 2) * 0.5);
        errors2.push_back(pow(Y[i] - ceres::exp(mgt*X[i] + cgt), 2) * 0.5);
    }

    std::ofstream fout("mc.txt");
    fout << m << " " << c << endl;
    fout.close();

    double Error1 = std::accumulate(errors1.begin(), errors1.end(), 0.0);
    double Error2 = std::accumulate(errors2.begin(), errors2.end(), 0.0);

    cout << "最小二乘法的结果并不是 m = 3, c = 0.75，实际上，　" << endl;
    cout << "最小二乘的cost function Error1 = " << Error1;
    cout << ", ground truth的cost function Error2 = " << Error2 << "， 比我们得到的解8.6051还要大\n";
}

