//
// Created by lightol on 3/7/19.
//

#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <sophus/so3.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <fstream>

using std::cout;
using std::endl;
using std::cerr;

bool LoadGtPoses(const std::string &poseFile, std::vector<Eigen::Matrix4d> &vGtTcws, bool bSTCC, int endID) {
    std::ifstream fTime(poseFile);
    if (!fTime.is_open()) {
        cerr << "open file " << poseFile << " failed." << endl;
        return false;
    }
    std::string line;
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    int nPose = 0;
    while (getline(fTime, line)) {
        if (nPose++ >= endID) {
            break;
        }
        if (bSTCC) {
            line = line.substr(line.find(' ') + 1);
        }

        std::stringstream ss(line);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> Twc(i, j);
            }
        }

        vGtTcws.push_back(Twc.inverse());
    }

    return true;
}

bool LoadLoopEdges(const std::string &poseFile, std::vector<Eigen::Matrix4d> &vTcw12s,
                   std::vector<std::pair<int, int> > &vID12s) {
    std::ifstream fin(poseFile);
    if (!fin.is_open()) {
        cerr << "open file " << poseFile << " failed." << endl;
        return false;
    }
    std::string line;
    Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
    int id1(0), id2(0);
    while (getline(fin, line)) {
        std::stringstream ss(line);

        ss >> id1 >> id2;
        vID12s.push_back(std::make_pair(id1, id2));

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> Tcw(i, j);
            }
        }

        vTcw12s.push_back(Tcw);
    }

    return true;
}

struct PoseError
{
    explicit PoseError(const Eigen::Matrix4d &_Tcw12): Tcw12(_Tcw12)
    {}

    template <typename T>  // 这个T是double或者Jet
    bool operator()(const T* const pKeyFrame1, const T* const pKeyFrame2, T* residual) const
    {
        // 注意这里的入参KeyFrame以及mapPoint等都要满足加法与乘法的封闭性，因为ceres的自动求导不可能还去帮你
        // 考虑R单位正交的特性，它只能是认为相机的位姿以及点的位置都是加乘法封闭的，所以我们不能给R而要转换成旋转向量给进来
        // Step0: 数据转换，把所有的数组转换为Eigen::Matrix
        // 在这里使用Eigen::Map<>方法，从data指针直接构造Eigen::Matrix，模版参数表示: const表示矩阵行列数固定，根据行列数从指针处读取固定数目的值构成Eigen::Matrix
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rotateVector1(pKeyFrame1);
        Eigen::Matrix<T, 4, 4> Tcw1 = Eigen::Matrix<T, 4, 4>::Identity();
        Eigen::AngleAxis<T> rv1;
        if (rotateVector1.norm() > T(0)) {
            rv1 = Eigen::AngleAxis<T>(rotateVector1.norm(), rotateVector1/rotateVector1.norm());
            Eigen::Matrix<T, 3, 3> R1(rv1);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> t1(pKeyFrame1+3);
            Tcw1.topLeftCorner(3, 3) = R1;
            Tcw1.topRightCorner(3, 1) = t1;
        }

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> rotateVector2(pKeyFrame2);
        Eigen::Matrix<T, 4, 4> Tcw2 = Eigen::Matrix<T, 4, 4>::Identity();
        Eigen::AngleAxis<T> rv2;
        if (rotateVector2.norm() > T(0)) {
            rv2 = Eigen::AngleAxis<T>(rotateVector2.norm(), rotateVector2/rotateVector2.norm());
            Eigen::Matrix<T, 3, 3> R2(rv2);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> t2(pKeyFrame2+3);
            Tcw2.topLeftCorner(3, 3) = R2;
            Tcw2.topRightCorner(3, 1) = t2;
        }

        // Step1: 将这个mapPoint变换到相机坐标系下，先选转再平移，KeyFrame的前3维是旋转向量， 后3维是平移t
        Eigen::Matrix<T, 4, 4> Tres = Tcw2.inverse() * Tcw12.template cast<T>() * Tcw1;
        Eigen::Matrix<T, 3, 1> tres = Tres.topRightCorner(3, 1);
//        Eigen::Matrix<T, 3, 1> tres = Eigen::Matrix<T, 3, 1>::Ones();
        residual[0] = tres.norm();
        Eigen::Matrix<T, 3, 3> Rres = Tres.topLeftCorner(3, 3);

        // 用四元数的第一个元素表示转角的大小
//        Eigen::Quaternion<T> Qua(Rres);
//        residual[2] = Qua.norm();


        return true;
    }

    const Eigen::Matrix4d Tcw12;
};

int main(int argc, char **argv) {
    std::string path_to_pose(argv[1]);
    std::string path_to_constraint(argv[2]);

    // Step0: Read in pose ground truth and loop constraints
    // step0.0: read pose ground truth
    int minImgIDs = 0;
    int nImages = 800;
    std::vector <Eigen::Matrix4d> vGtTcws;
    vGtTcws.reserve(nImages);
    if (!LoadGtPoses(path_to_pose, vGtTcws, true, nImages)) {
        cerr << "Load Pose Ground truth file failed" << endl;
        return 1;
    }

    //step0.1: read loop edge
    std::vector <Eigen::Matrix4d> vTcw12s;
    std::vector <std::pair<int, int>> vID12s;
    vTcw12s.reserve(10 * nImages);
    vID12s.reserve(10 * nImages);
    if (!LoadLoopEdges(path_to_constraint, vTcw12s, vID12s)) {
        cerr << "Load loop edge file failed" << endl;
        return 1;
    }

    // Step1: ceres求解
    double Rts[6 * nImages];
    double* ptr = &Rts[0];
    for (int i = 0; i < nImages; ++i) {
        Eigen::Matrix4d Tcw = vGtTcws[i];
        Eigen::Matrix3d R = Tcw.topLeftCorner(3, 3);
        Sophus::SO3 so3(R);
        Eigen::Vector3d rv = so3.log();
        for (int j = 0; j < 3; ++j) {
            Rts[6*i + j] = rv[j];
        }
        Eigen::Vector3d t = Tcw.topRightCorner(3, 1);
        for (int j = 0; j < 3; ++j) {
            Rts[6*i + j + 3] = t[j];
        }
    }

    double preRts[6 * nImages];
    for (int i = 0; i < 6 * nImages; ++i) {
        preRts[i] = Rts[i];
    }

    ceres::Problem problem;
    for (int i = minImgIDs; i < nImages - 1; ++i) {
        Eigen::Matrix4d Tcw12 = vGtTcws[i+1] * vGtTcws[i].inverse();
        ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<PoseError, 1, 6, 6>(
                new PoseError(Tcw12));
        // 第一个参数是代价函数，第二个参数是损失函数，后面的都是cost_function的待优化变量的指针，
        // 并且待优化变量必须是double类型的，也就是不管是9维的R还是3维的t都必须转化为double输入进去，
        // 它们的维度在构造cost_function的模板里面给定了，入参给指针，也就是地址起点，根据模板参数去确定取出几个double
        problem.AddResidualBlock(pCostFunction, new ceres::HuberLoss(5.991), Rts+6*i, Rts+6*(i+1));
    }

    // 构造loop edges
//    for (int i = 0; i < vID12s.size(); ++i) {
//        Eigen::Matrix4d Tcw12 = vTcw12s[i];
//        auto ids = vID12s[i];
//        if (ids.first < minImgIDs || ids.first > nImages || ids.second < minImgIDs || ids.first > nImages) {
//            continue;
//        }
//
//        ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<PoseError, 1, 6, 6>(
//                new PoseError(Tcw12));
//        // 第一个参数是代价函数，第二个参数是损失函数，后面的都是cost_function的待优化变量的指针，
//        // 并且待优化变量必须是double类型的，也就是不管是9维的R还是3维的t都必须转化为double输入进去，
//        // 它们的维度在构造cost_function的模板里面给定了，入参给指针，也就是地址起点，根据模板参数去确定取出几个double
//        problem.AddResidualBlock(pCostFunction, new ceres::HuberLoss(5.991), Rts+6*ids.second, Rts+6*ids.first);
//    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 60;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";

    for (int i = minImgIDs; i < nImages - 1; ++i) {
        for (int j = 0; j < 6; ++j) {
            cout << Rts[6*i + j] - preRts[6*i + j] << endl;
        }
    }

}