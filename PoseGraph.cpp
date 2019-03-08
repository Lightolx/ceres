//
// Created by lightol on 3/7/19.
//

#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <sophus/so3.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>
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

        cout << Twc.topLeftCorner(3, 3).determinant() << endl;
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

struct Pose3d {
    Eigen::Vector3d p;
    Eigen::Quaterniond q;

    // The name of the data type in the g2o file format.
    static std::string name() {
        return "VERTEX_SE3:QUAT";
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::map<int, Pose3d, std::less<int>,
        Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
        MapOfPoses;

typedef std::vector<Pose3d, Eigen::aligned_allocator<Pose3d> >
        VectorOfPoses;

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
    int id_begin;
    int id_end;

    // The transformation that represents the pose of the end frame E w.r.t. the
    // begin frame B. In other words, it transforms a vector in the E frame to
    // the B frame.
    Pose3d t_be;

    // The inverse of the covariance matrix for the measurement. The order of the
    // entries are x, y, z, delta orientation.
    Eigen::Matrix<double, 6, 6> information;

    // The name of the data type in the g2o file format.
    static std::string name() {
        return "EDGE_SE3:QUAT";
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
        VectorOfConstraints;

class PoseGraph3dErrorTerm {
public:
    PoseGraph3dErrorTerm(const Pose3d& t_ab_measured,
                         const Eigen::Matrix<double, 6, 6>& sqrt_information)
            : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {}

    template <typename T>
    bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                    const T* const p_b_ptr, const T* const q_b_ptr,
                    T* residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);

        Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

        // Compute the relative transformation between the two frames.
        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();  // 相当于Matrix.inverse()
        Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

        // Represent the displacement between the two frames in the A frame.
        Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

        // Compute the error between the two orientation estimates.
        Eigen::Quaternion<T> delta_q =
                t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

        // the element order for Ceres's quaternion is [q_w, q_x, q_y, q_z] where as Eigen's quaternion
        // is [q_x, q_y, q_z, q_w].

        // Compute the residuals.
        // [ position         ]   [ delta_p          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) =
                p_ab_estimated - t_ab_measured_.p.template cast<T>();
        residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

        // Scale the residuals by the measurement uncertainty.
        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

        return true;
    }

    static ceres::CostFunction* Create(
            const Pose3d& t_ab_measured,
            const Eigen::Matrix<double, 6, 6>& sqrt_information) {
        return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
                new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Pose3d t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                              VectorOfPoses &poses, ceres::Problem* problem) {
    CHECK(problem != NULL);
    if (constraints.empty()) {
        LOG(INFO) << "No constraints, no problem to optimize.";
        return;
    }

    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization* quaternion_local_parameterization =
            new ceres::EigenQuaternionParameterization;

    for (VectorOfConstraints::const_iterator constraints_iter =
            constraints.begin();
         constraints_iter != constraints.end(); ++constraints_iter) {
        const Constraint3d& constraint = *constraints_iter;
        int id1 = constraint.id_begin;
        int id2 = constraint.id_end;

        const Eigen::Matrix<double, 6, 6> sqrt_information =
                constraint.information.llt().matrixL();
        // Ceres will take ownership of the pointer.
        ceres::CostFunction* cost_function =
                PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

        problem->AddResidualBlock(cost_function, loss_function,
                                  poses[id1].p.data(),
                                  poses[id1].q.coeffs().data(),
                                  poses[id2].p.data(),
                                  poses[id2].q.coeffs().data());

        // 设定这两个地址所指向的变量为Quaternion流形, 优化时可以只在流形的某一个方向迭代
        problem->SetParameterization(poses[id1].q.coeffs().data(),
                                     quaternion_local_parameterization);
        problem->SetParameterization(poses[id2].q.coeffs().data(),
                                     quaternion_local_parameterization);
    }

    // The pose graph optimization problem has six DOFs that are not fully
    // constrained. This is typically referred to as gauge freedom. You can apply
    // a rigid body transformation to all the nodes and the optimization problem
    // will still have the exact same cost. The Levenberg-Marquardt algorithm has
    // internal damping which mitigates this issue, but it is better to properly
    // constrain the gauge freedom. This can be done by setting one of the poses
    // as constant so the optimizer cannot change it.
    // 把第一个pose给fix住
    problem->SetParameterBlockConstant(poses[0].p.data());
    problem->SetParameterBlockConstant(poses[0].q.coeffs().data());

}

int main(int argc, char **argv) {
    std::string path_to_pose(argv[1]);
    std::string path_to_constraint(argv[2]);

    // Step0: Read in pose ground truth and loop constraints
    // step0.1: read pose ground truth
    int minImgIDs = 0;
    int nImages = 6500;
    std::vector <Eigen::Matrix4d> vGtTcws;
    vGtTcws.reserve(nImages);
    if (!LoadGtPoses(path_to_pose, vGtTcws, true, nImages)) {
        cerr << "Load Pose Ground truth file failed" << endl;
        return 1;
    }
    cout << vGtTcws[0].topLeftCorner(3, 3).determinant() << endl;

    //step0.2: read loop edge
    std::vector <Eigen::Matrix4d> vTcw12s;
    std::vector <std::pair<int, int>> vID12s;
    vTcw12s.reserve(10 * nImages);
    vID12s.reserve(10 * nImages);
    if (!LoadLoopEdges(path_to_constraint, vTcw12s, vID12s)) {
        cerr << "Load loop edge file failed" << endl;
        return 1;
    }

    // ceres求解
    // Step1: 构建每个KeyFrame的poses, 注意用四元数表示
    VectorOfPoses poses;
    poses.resize(nImages);
    VectorOfPoses poses0;
    poses0.resize(nImages);
    for (int i = 0; i < nImages; ++i) {
        Pose3d pose3d;
        Eigen::Matrix4d Tcw = vGtTcws[i];
        Eigen::Matrix3d R = Tcw.topLeftCorner(3, 3);
        pose3d.q = Eigen::Quaterniond(R);
        pose3d.p = Tcw.topRightCorner(3, 1);
        poses[i] = pose3d;
        poses0[i] = pose3d;
//        poses0[i] = pose3d;

//        if (i != 0) {
//            poses[i].p += 0.1*Eigen::Vector3d::Random();
//        }
        {
            Eigen::Matrix4d Twc = Tcw.inverse();
            Eigen::Vector3d t0 = Twc.topRightCorner(3, 1);
            Eigen::Matrix3d R = Tcw.topLeftCorner(3, 3);
            cout << "R.det = " << R.determinant() << endl;
            cout << "R.trans - R.inv =\n" << R.transpose() - R.inverse() << endl << endl;
            Eigen::Vector3d t1 = -R.inverse() * Tcw.topRightCorner(3, 1);
//            cout << "t0 is " << t0.transpose() << endl;
//            cout << "t1 is " << t1.transpose() << endl << endl;
        }

        {
//            Eigen::Matrix4d Twc = Tcw.inverse();
//            Eigen::Vector3d t0 = Twc.topRightCorner(3, 1);
//            Eigen::Matrix3d R(pose3d.q);
//            Eigen::Vector3d t1 = -R.transpose() * pose3d.p;
//            cout << "t0 is " << t0.transpose() << endl;
//            cout << "t1 is " << t1.transpose() << endl << endl;
        }
    }

    // Step2: 构建normal edge, 也就是根据gtPose算出的前后帧之间的相对变换
    VectorOfConstraints constraints;    // 所有的graph edge
    for (int i = 0; i < nImages; ++i) {
        Constraint3d constraint3d;

        Pose3d pose1 = poses[i];
        Pose3d pose2 = poses[i+1];

        if (i == nImages-1) {
            pose1 = poses[nImages-1];
            pose2 = poses[0];
        }

        Pose3d pose12;
        pose12.q = pose1.q.conjugate() * pose2.q;
        pose12.p = pose1.q.conjugate() * (pose2.p - pose1.p);

        // 这里告诉我们, 从Eigen::Matrix转到Eigen::Quaternion是有精度损失的,
        // 既然在ceres里要用四元数, 那么在这里也早早地转成四元数
        /*
        cout << "relative R is\n" << pose3d.q.matrix() << endl;
        cout << "relative t is " << pose3d.p.transpose() << endl;

        Eigen::Matrix4d Tcw1 = vGtTcws[i];
        Eigen::Matrix4d Tcw2 = vGtTcws[i+1];
        Eigen::Matrix4d Tcw12 = Tcw1.inverse() * Tcw2;
        Eigen::Matrix3d R = Tcw12.topLeftCorner(3, 3);
        Eigen::Quaterniond quaterniond(R);
        cout << "while by Tcw\nrelative R is\n" << quaterniond.matrix() << endl;
        cout << "relative t is " << Tcw12.topRightCorner(3, 1).transpose() << endl << endl;
         */

        constraint3d.id_begin = i;
        constraint3d.id_end = (i+1 < nImages) ? i+1 : 0;  // 如果溢出, 则说明与第一帧回环
        constraint3d.t_be = pose12;
        constraint3d.information = Eigen::Matrix<double, 6, 6>::Identity();
        constraints.push_back(constraint3d);
    }

    // Step3: 构造loop edge, 根据回环检测判定出的当前帧相对于回环帧的相对变换
    int nLoopEdges = vID12s.size();
    for (int i = 0; i < 0; ++i) {
        int id1 = vID12s[i].first;
        int id2 = vID12s[i].second;
        Eigen::Matrix4d Tcw2 = vTcw12s[i];
        Eigen::Matrix4d gtTcw2 = vGtTcws[id2];
        cout << "gt Tcw2 is\n" << gtTcw2 << endl;
        cout << "while measure Tcw2 is\n" << Tcw2 << endl;

        // step3.1: 先把Tcw2转换成四元数
        Pose3d pose2;
        Eigen::Matrix3d R = Tcw2.topLeftCorner(3, 3);
        pose2.q = Eigen::Quaterniond(R);
        pose2.p = Tcw2.topRightCorner(3, 1);

        // step3.2: 求出pose1到pose2的相对变换
        Pose3d pose1 = poses[id1];

        Pose3d pose12;
        pose12.q = pose1.q.conjugate() * pose2.q;
        pose12.p = pose1.q.conjugate() * (pose2.p - pose1.p);

        // step3.3: 构造constraint
        Constraint3d constraint3d;

        constraint3d.id_begin = id1;
        constraint3d.id_end = id2;
        constraint3d.t_be = pose12;
        constraint3d.information = Eigen::Matrix<double, 6, 6>::Identity();
        constraints.push_back(constraint3d);
    }

    // step3.5: 优化之前检测一下total residual

    int nConstraints = constraints.size();
    double errors = 0.0;
    for (int i = 0; i < nConstraints; ++i) {
        Constraint3d constraint3d = constraints[i];
        int id1 = constraint3d.id_begin;
        int id2 = constraint3d.id_end;
        Pose3d pose3d = constraint3d.t_be;
        Pose3d pose1 = poses[id1];
        Pose3d pose2 = poses[id2];

        auto q_estimate = pose1.q.conjugate() * pose2.q;
        auto q_measure = pose3d.q;
        auto q_error = q_measure * q_estimate.conjugate();

        auto p_estimate = pose1.q.conjugate() * (pose2.p - pose1.p);
        auto p_measure = pose3d.p;
        auto p_error = p_estimate - p_measure;

        Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
        error.topRows(3) = p_error;
        error.bottomRows(3) = 2 * q_error.vec();
//        cout << "error is " << error.transpose() << endl;
        errors += 0.5 * error.transpose() * error;
    }

    cout << "before optimize, errors = " << errors << endl;

    /*
    // Step3.6: 随机加入一些回环拉回pose
    for (int nLoops = 0; nLoops < std::atoi(argv[3]); ++nLoops) {
        int i = rand() % nImages;
        int j = rand() % nImages;
        while (i == j) {
            j = rand() % nImages;
        }

        Constraint3d constraint3d;

        Pose3d pose1 = poses0[i];
        Pose3d pose2 = poses0[j];

        Pose3d pose3d;
        pose3d.q = pose1.q.conjugate() * pose2.q;
        pose3d.p = pose1.q.conjugate() * (pose2.p - pose1.p);

        constraint3d.id_begin = i;
        constraint3d.id_end = j;
        constraint3d.t_be = pose3d;
        constraint3d.information = Eigen::Matrix<double, 6, 6>::Identity();
        constraints.push_back(constraint3d);
    }*/

    // Step4: 构造ceres::Problem并优化
    ceres::Problem problem;
    BuildOptimizationProblem(constraints, poses, &problem);

    ceres::Solver::Options options;
    options.max_num_iterations = 3;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

//    std::cout << summary.FullReport() << '\n';

    // Step5: 最后输出一下误差
    double sumErrors = 0.0;
    for (int i = 0; i < nImages; ++i) {
        Eigen::Matrix4d preTcw0 = vGtTcws[i].inverse();
        Eigen::Matrix3d preR0 = preTcw0.topLeftCorner(3, 3);
        Eigen::Vector3d pret0 = preTcw0.topRightCorner(3, 1);
        pret0 = -preR0.transpose() * pret0;
        cout << pret0.transpose() << endl;

        Pose3d pose0 = poses0[i];
        Eigen::Matrix3d R0(pose0.q);
        Eigen::Vector3d t0 = pose0.p;
        t0 = -R0.transpose() * t0;
        cout << t0 << endl;

        Pose3d pose1 = poses[i];
        Eigen::Matrix3d R1(pose1.q);
        Eigen::Vector3d t1 = pose1.p;
        t1 = -R1.transpose() * t1;


        cout << "id: " << i << ", diff = " << (t1 - t0).transpose() << endl;
        sumErrors += (t1 - t0).norm();
    }
//    cout << "errors = " << sumErrors << endl;
}