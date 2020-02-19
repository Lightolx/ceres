//
// Created by lightol on 2019/10/25.
//

#include <vector>


#include <eigen3/Eigen/Eigen>

#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>

#include <sophus/so3.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct RegistrationError {
    RegistrationError(const Eigen::Vector3d &_p, const Eigen::Vector3d &_q):p(_p), q(_q) {}

    template<typename T>
    bool operator()(const T* const ksi, T* residual) const {
        T R[3*3];
        ceres::AngleAxisToRotationMatrix(ksi, R);
        Eigen::Matrix<T, 3, 3> Rcw_inv = Eigen::Matrix<T, 3, 3>::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Rcw_inv(i, j) = R[3*i + j];
            }
        }
        // ceres::AngleAxisToRatationMatrix()的结果是按照列排序的
        Eigen::Matrix<T, 3, 3> Rcw = Rcw_inv.transpose();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> tcw(ksi+3);

        residual[0] = (q.template cast<T>() - (Rcw * p.template cast<T>() + tcw)).norm();

        return true;
    }

    const Eigen::Vector3d p;
    const Eigen::Vector3d q;
};

int main() {
    pcl::PointCloud<pcl::PointXYZI> cloud_roadmark;
    pcl::io::loadPLYFile("/home/lightol/Desktop/35arrow/arrow.ply", cloud_roadmark);
    PointCloudT::Ptr cloud_template(new pcl::PointCloud<PointT>);
    pcl::io::loadPLYFile("/home/lightol/Desktop/35arrow/template.ply", *cloud_template);
    // Step1: Generate the template PC and roadmark PC
    int num_template_pts = cloud_template->points.size();
    std::vector<Eigen::Vector3d> template_pts(num_template_pts);
    for (int i = 0; i < num_template_pts; ++i) {
        const auto &pt = cloud_template->points[i];
        template_pts[i] = Eigen::Vector3d(pt.x, pt.y, pt.z);
    }

    int num_roadmark_pts = cloud_roadmark.points.size();
    std::vector<Eigen::Vector3d> roadmark_pts(num_roadmark_pts);
    std::vector<float> vIntensity(num_roadmark_pts);
    for (int i = 0; i < num_roadmark_pts; ++i) {
        const auto &pt = cloud_roadmark.points[i];
        roadmark_pts[i] = Eigen::Vector3d(pt.x, pt.y, pt.z);
        vIntensity[i] = pt.intensity;
//        vIntensity[i] = 1;
    }
    // roadmark点云中每个点都有不同的权重
    float total_intensity = std::accumulate(vIntensity.begin(), vIntensity.end(), 0.0);
    for (float &intensity : vIntensity) {
        intensity /= total_intensity;
    }

    // Step2: 优化6自由度的pose，使roadmarker旋转到与template比较贴合，也就是一个加权ICP
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_template);

    // step2.3: R用旋转向量表示
    double ksi[6];
    for (int i = 0; i < 6; ++i) {
        ksi[i] = 0;
    }

    // Step3: ICP the two point cloud
    for (int i = 0; i < 600; ++i) {
        Eigen::Map<const Eigen::Vector3d> fai(ksi);
        Eigen::Matrix3d Rwc = Sophus::SO3::exp(fai).matrix();
        Eigen::Map<const Eigen::Vector3d> twc(ksi+3);

        // Step2.1: for each pt in source_pts, find its nearest pt in target_pts
        PointT searchPoint;
        int K = 1;
        std::vector<int> pointIdx(K);
        std::vector<float> pointDist(K);
        Eigen::Vector3d nearestPt = Eigen::Vector3d::Zero();

        ceres::Problem problem;
        for (const auto &point : roadmark_pts) {
            Eigen::Vector3d pt = Rwc * point + twc;
            searchPoint.x = pt.x();
            searchPoint.y = pt.y();
            searchPoint.z = pt.z();

            if (kdtree.nearestKSearch(searchPoint, K, pointIdx, pointDist) > 0) {
                nearestPt = template_pts[pointIdx[0]];
                ceres::CostFunction* pCostFunction = new ceres::AutoDiffCostFunction<RegistrationError, 1, 6>(new RegistrationError(point, nearestPt));
                problem.AddResidualBlock(pCostFunction, new ceres::CauchyLoss(0.5), ksi);
            }
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
//        std::cout << summary.FullReport() << std::endl;
    }

    // Step3: Output the ICP result
    Eigen::Map<const Eigen::Vector3d> fai(ksi);
    Eigen::Matrix3d Rwc = Sophus::SO3::exp(fai).matrix();
    Eigen::Map<const Eigen::Vector3d> twc(ksi+3);
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.topLeftCorner(3, 3) = Rwc;
    Twc.topRightCorner(3, 1) = twc;

    pcl::PointCloud<pcl::PointXYZI> cloud_roadmark_transed;
    pcl::transformPointCloud(cloud_roadmark, cloud_roadmark_transed, Twc);
    pcl::io::savePLYFile("/home/lightol/Desktop/35arrow/roadmark_transed.ply", cloud_roadmark_transed);

    return 0;
}