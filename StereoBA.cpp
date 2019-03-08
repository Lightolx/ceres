//
// Created by lightol on 2/23/19.
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

struct StereoReprojectError
{
    explicit StereoReprojectError(double _u, double _v, double _ru, Eigen::Matrix3d _K): u(_u), v(_v), rv(_ru), K(_K)
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
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pw(pMapPoint);

        // Step1: 将这个mapPoint变换到相机坐标系下，先选转再平移，KeyFrame的前3维是旋转向量， 后3维是平移t
        Eigen::Matrix<T, 3, 1> Pc = R*Pw + t;
        T depth = Pc[2];
        Pc /= Pc[2];  // 转换到归一化平面

        // Step2: 转换到uv坐标系
        Eigen::Matrix<T, 3, 1> Puv = K.template cast<T>() * Pc;

        // Step3: 归一化平面转换到成像平面上，小孔成像投影模型计算在成像平面上的(u,v)
        T predict_u = pKeyFrame[6] * Puv[0];
        T predict_v = pKeyFrame[6] * Puv[1];

        // Step4: 最终计算残差，预测减去观测
        residual[0] = Puv[0] - u;
        residual[1] = Puv[1] - v;
        residual[2] = Puv[0] - bf / depth - ru;  // 右视图上的残差

        return true;
    }

    const double u;
    const double v;
    const double ru;
    const Eigen::Matrix3d K;
    const double bf;
};