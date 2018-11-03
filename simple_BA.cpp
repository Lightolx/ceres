#include <iostream>
#include <eigen3/Eigen/Eigen>
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
    bool operator()(const T* const KeyFrame, const T* const mapPoint, T* residual) const
    {
        T p[3];
        // 将这个mapPoint变换到相机坐标系下，先选转再平移，KeyFrame的前3维是旋转向量
        // step1: 旋转R*p，这个函数的意思是按照keyframe所代表的旋转向量将mapPoint进行旋转，结果保存在p中
        ceres::AngleAxisRotatePoint(KeyFrame, mapPoint, p);
        // step2: 平移R*p + t，KeyFrame的后3维是平移t，直接相加即可
        p[0] += KeyFrame[3]; p[1] += KeyFrame[4]; p[2] += KeyFrame[5];

        // 去畸变，keyFrame的第7个参数是焦距fx = fy，并且假设cx = cy = 0，第8,9两个参数是去畸变系数
        T Xc = -p[0] / p[2];
        T Yc = -p[1] / p[2];
        T r2 = Xc*Xc + Yc*Yc;
        // 去畸变后在归一化平面上的坐标
        T XcUndist = Xc*(T(1) + KeyFrame[7]*r2 + KeyFrame[8]*r2*r2);
        T YcUndist = Yc*(T(1) + KeyFrame[7]*r2 + KeyFrame[8]*r2*r2);

        // 小孔成像投影模型计算在成像平面上的(u,v)
        T predict_u = KeyFrame[6] * Xc;
        T predict_v = KeyFrame[6] * Yc;

        // 最终计算残差
        residual[0] = u_ - predict_u;
        residual[1] = v_ - predict_v;

        return true;
    }

    const double u_;
    const double v_;
};

//class BAdata
//{
//public:
//    BAdata(const std::string &str)
//    {
//        LoadFile(str);
//    }
//
//    ~BAdata()
//    {
//        delete[](KFids);
//        delete[](MPids);
//        delete[](observations);
//        delete[](KFs);
//        delete[](Mps);
//    }
//
//private:
//    template <typename T>
//    void FscanfOrDie(FILE *fptr, const char *format, T* value)
//    {
//        int numScan = fscanf(fptr, format, value);
//
//        if (numScan != 1)
//        {
//            std::cerr << "cannot load data file, please check if it exists" << endl;
//            std::abort();
//        }
//    }
//
//    void LoadFile(const std::string &str)
//    {
//        FILE* fptr = fopen(str.c_str(), "r");
//        if (!fptr)
//        {
//            std::cerr << "cannot load data file at " << str << ", please check if it exists" << endl;
//            std::abort();
//        }
//
//        // step1: 读入第一行，keyframe, mapPoint以及observation的个数
//        FscanfOrDie(fptr, "%d", &nKFs);
//        FscanfOrDie(fptr, "%d", &nMPs);
//        FscanfOrDie(fptr, "%d", &nObservations);
//
//        // step2: 读取每个observation中，KF的id以及mapPoint的id以及在KF上观测到的这个mapPoint的(u,v)坐标
//        KFids = new int[nObservations];
//        MPids = new int[nObservations];
//        observations = new double[nObservations*2];
//        for (int i = 0; i < nObservations; ++i)
//        {
//            FscanfOrDie(fptr, "%d", KFids+i);
//            FscanfOrDie(fptr, "%d", MPids+i);
//            FscanfOrDie(fptr, "%lf", observations+2*i);
//            FscanfOrDie(fptr, "%lf", observations+2*i+1);
//        }
//
//        // step3: 读取每个keyframe的pose的初始值
//        KFs = new double*[nKFs];
//        for (int i = 0; i < nKFs; ++i)
//        {
//            KFs[i] = new double[9];
//            for (int j = 0; j < 9; ++j)
//            {
//                FscanfOrDie(fptr, "%lf", KFs[i]+j);
//            }
//        }
//
//        // step4: 读取每个mapPoint的世界坐标的初始值
//        Mps = new double*[nMPs];
//        for (int i = 0; i < nMPs; ++i)
//        {
//            Mps[i] = new double[3];
//            for (int j = 0; j < 3; ++j)
//            {
//                FscanfOrDie(fptr, "lf%", Mps[i]+j);
//            }
//        }
//    }
//
////private:
//public:
//    int nKFs;
//    int nMPs;
//    int nObservations;
//    int* KFids;
//    int* MPids;
//    double* observations;
//    double** KFs;
//    double** Mps;
//};

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
        problem.AddResidualBlock(pCostFunction, nullptr, BA.pKFs + 9*KFid, BA.pMPs + 3*MPid);
//        double u = BA.vObservations[i][0];
//        double v = BA.vObservations[i][1];
//        double kf = *(BA.pKFs + 9*KFid);
//        double mp = *(BA.pMPs + 3*MPid);
//        cout << u << " " << v << " " << kf << " " << mp << endl;
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

