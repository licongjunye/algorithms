#include<iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 目标函数 F_fun3
double F_fun3(double x, double y, double z) {
    return (x - 2000.5) * (x - 2000.5) + (y + 155.8) * (y + 155.8) + (z - 10.25) * (z - 10.25);
}

// 计算残差向量
Mat computeResiduals(const Mat &x) {
    Mat r(1, 1, CV_64F);
    double x0 = x.at<double>(0, 0);
    double y0 = x.at<double>(1, 0);
    double z0 = x.at<double>(2, 0);

    // 计算残差
    r.at<double>(0, 0) = F_fun3(x0, y0, z0);
    return r;
}

// 计算雅可比矩阵
Mat computeJacobian(const Mat &x) {
    double EPS = 1e-6;
    int n = x.rows; // 参数的数量
    Mat r = computeResiduals(x);
    int m = r.rows; // 残差的数量

    Mat J(m, n, CV_64F); // 创建一个 m 行 n 列的雅可比矩阵

    for (int i = 0; i < n; ++i) { 
        // 创建一个偏移量向量
        Mat x_plus_eps = x.clone();
        x_plus_eps.at<double>(i, 0) += EPS;
        // 计算偏移后的残差
        Mat r_plus_eps = computeResiduals(x_plus_eps);

        // 计算雅可比矩阵的每一列
        for (int j = 0; j < m; ++j) {
            J.at<double>(j, i) = (r_plus_eps.at<double>(j, 0) - r.at<double>(j, 0)) / EPS;
        }
    }

    return J;
}

void LM_optimize(double &x0, double &y0, double &z0)
{
    int iter = 200000; // 迭代次数
    double e1 = 1e-5;  // 误差限
    double e2 = 1e-5; 
    double u = 0.000001;    // 初始 u 值
    Mat last_xk;
    Mat xk = (Mat_<double>(3, 1) << x0, y0, z0); // 初始化输入参数
    Mat I = Mat::eye(xk.rows, xk.rows, CV_64F);  // 单位矩阵
    double f1, f2;

    for(int i = 0; i < iter; i++)
    {
        // 计算雅可比矩阵 J
        Mat J = computeJacobian(xk);

        // 计算残差向量 r
        Mat r = computeResiduals(xk);
        
        // 计算 J^T
        Mat J_T = J.t();

        // 计算 J^T * J
        Mat H = J_T * J;
        H = H + u * I;

        // 计算 J^T * r
        Mat g = J_T * r;

        // 计算 (J^T * J + u * I)^-1
        Mat H_inv = H.inv();

        // 计算更新量 Δx
        Mat delta_x = -H_inv * g;
        last_xk = xk.clone();
        f1 = F_fun3(xk.at<double>(0, 0), xk.at<double>(1, 0), xk.at<double>(2, 0));   // 计算 f(Xk)

        // 更新变量 xk
        xk = xk + delta_x;
        f2 = F_fun3(xk.at<double>(0, 0), xk.at<double>(1, 0), xk.at<double>(2, 0));  // 计算更新后的 f(Xk+1)

        if(f2 >= f1 * 1.5)   // 如果 fk+1 > fk，说明当前逼近方向出现偏差，导致跳过了最优点，需要通过增大 u 值来减小步长
        {
            u *= 1.15;    // 增大 u 值
            xk = last_xk.clone();
        }
        else if(f2 < f1)  // 如果 fk+1 < fk，说明当前步长合适，可以通过减小 u 值来增大步长，加快收敛速度
        {
            u *= f2 / f1;   // 减小 u 值
        }

        printf("i=%d, f1=%f, f2=%f, u=%f\n", i, f1, f2, u);
        // 如果输入参数的更新量很小，或者目标函数值变化很小，则认为寻找到最优参数，停止迭代
        if(norm(delta_x, NORM_L2) < e1 && abs(f1-f2) < e2)
            break;
    }

    x0 = xk.at<double>(0, 0);
    y0 = xk.at<double>(1, 0);
    z0 = xk.at<double>(2, 0);
}

int main(int argc, char const *argv[])
{
    double x = 100, y = 100, z = 100;
    LM_optimize(x, y, z);
    std::cout << "x=" << x << "\t y=" << y << "\t z=" << z << std::endl;
    return 0;
}