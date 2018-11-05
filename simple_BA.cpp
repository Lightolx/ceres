#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <sophus/so3.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <fstream>

using std::cout;
using std::endl;

struct ReprojectError
{
    explicit ReprojectError(double u, double v): u_(u), v_(v)
    {}

    template <typename T>  // 这个T是double或者Jet
    bool operator()(const T* const pKeyFrame, const T* const pMapPoint, T* residual) const
    {
        // 注意这里的入参KeyFrame以及mapPoint等都要满足加法与乘法的封闭性，因为ceres的自动求导不可能还去帮你考虑R单位正交的特性，它只能是认为相机的位姿以及点的位置都是加乘法封闭的，所以我们不能给R而要转换成旋转向量给进来
        // Step0: 数据转换，把所有的数组转换为Eigen::Matrix
        // 在这里使用Eigen::Map<>方法，从data指针直接构造Eigen::Matrix，模版参数表示 const表示矩阵行列数固定，根据行列数从指针处读取固定数目的值构成Eigen::Matrix
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rotateVector(pKeyFrame);
        Eigen::AngleAxis<T> rv(rotateVector.norm(), rotateVector/rotateVector.norm());
        Eigen::Matrix<T, 3, 3> R(rv);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(pKeyFrame+3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Xw(pMapPoint);

        // Step1: 将这个mapPoint变换到相机坐标系下，先选转再平移，KeyFrame的前3维是旋转向量， 后3维是平移t
        Eigen::Matrix<T, 3, 1> Xc = R*Xw + t;
        Xc /= -Xc[2];  // 转换到归一化平面

        // Step2: 去畸变，keyFrame的第7个参数是焦距fx = fy，并且假设cx = cy = 0，第8,9两个参数是去畸变系数
        T r2 = Xc[0]*Xc[0] + Xc[1]*Xc[1];
        Xc *= T(1) + pKeyFrame[7]*r2 + pKeyFrame[8]*r2*r2;

        // Step3: 归一化平面转换到成像平面上，小孔成像投影模型计算在成像平面上的(u,v)
        T predict_u = pKeyFrame[6] * Xc[0];
        T predict_v = pKeyFrame[6] * Xc[1];

        // 最终计算残差，预测减去观测
        residual[0] = predict_u - u_;
        residual[1] = predict_v - v_;

        return true;
    }

    const double u_;
    const double v_;
};

class BAdata
{
public:
    BAdata(const std::string &str)
    {
        LoadFile(str);
    }

    ~BAdata()
    {
        delete[](pKFs);
        delete[](pMPs);
    }

    int nKFs;
    int nMPs;
    int nObservations;
    std::vector<std::pair<int, int> > vKFid_MPid;
    std::vector<Eigen::Vector2d> vObservations;  // 能直接观测到的量就是mapPoint在图像上的(u,v)坐标
    double* pKFs;  // nKFs*9的二维数组
    double* pMPs;  // nMPs*3的二维数组

private:
    void LoadFile(const std::string &str)
    {
        std::ifstream fin(str);
        std::string ptline;

        // step1: 读入第一行，keyframe, mapPoint以及observation的个数
        getline(fin, ptline);
        std::stringstream ss0(ptline);
        ss0 >> nKFs >> nMPs >> nObservations;

        // step2: 读取每个observation中，KF的id以及mapPoint的id以及在KF上观测到的这个mapPoint的(u,v)坐标
        vKFid_MPid.resize(nObservations);
        vObservations.resize(nObservations);
        for (int i = 0; i < nObservations; ++i)
        {
            getline(fin, ptline);
            std::stringstream ss(ptline);
            ss >> vKFid_MPid[i].first >> vKFid_MPid[i].second >> vObservations[i][0] >> vObservations[i][1];
        }

        // step3: 读取每个keyframe的pose的初始值
        pKFs = new double[9*nKFs];
        for (int i = 0; i < 9*nKFs; ++i)
        {
            getline(fin, ptline);
            std::stringstream ss(ptline);

            ss >> *(pKFs + i);
        }

        // step4: 读取每个mapPoint的世界坐标的初始值
        pMPs = new double[3*nMPs];
        for (int i = 0; i < 3*nMPs; ++i)
        {
            getline(fin, ptline);
            std::stringstream ss(ptline);

            ss >> *(pMPs + i);
        }
    }

};

int main()
{
    // Step0: Read in data
    BAdata BA("BA_data.txt");

    // Step1: 添加各个残差块，residualBlock，每一个observation就能构成一个残差块
    // one term is added to the objective function per observation. 这句话可以这样理解：
    // 对于空间中的任意一个mapPoint，相机在某个地方拍它的一张照片，看它出现在照片中的什么位置就是一次observation，
    // 这就和一条未知的曲线，我在x轴上任意一点画一条平行于y轴的直线，看它与曲线交点的y坐标是多少一样，也是一次observation．
    // 都是借由我们建立的模型的predict与在这次观测中我们得到的observation的差值，来评价我们建立的模型对观测数据的符合程度．
    // 只不过前者是相机投影模型，后者是指数曲线模型，其本质都是一样的，最后都是要慢慢修正我们模型里面各个参数的值
    ceres::Problem problem;
    for (int i = 0; i < BA.nObservations; ++i)
    {
        ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<ReprojectError, 2, 9, 3>(
                new ReprojectError(BA.vObservations[i][0], BA.vObservations[i][1]));
        int KFid = BA.vKFid_MPid[i].first;
        int MPid = BA.vKFid_MPid[i].second;
        // 第一个参数是代价函数，第二个参数是损失函数，后面的都是cost_function的待优化变量的指针，
        // 并且待优化变量必须是double类型的，也就是不管是9维的R还是3维的t都必须转化为double输入进去，
        // 它们的维度在构造cost_function的模板里面给定了，入参给指针，也就是地址起点，根据模板参数去确定取出几个double
        problem.AddResidualBlock(pCostFunction, nullptr, BA.pKFs + 9*KFid, BA.pMPs + 3*MPid);
    }

    // Step2: Solve it
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    return 0;
}

