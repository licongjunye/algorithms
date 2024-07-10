      
#include <iostream>
using namespace std; 

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

int main( int argc, char** argv )
{
    double dt = 0.5;
    // 状态量x
    Eigen::Vector2d x = Eigen::Vector2d::Zero();

    Eigen::Matrix2d F; //State Transition Matrix
    F << 1, dt,
         0, 1;

    double u[3] = {3, 2, 1};    //输入：加速度 Input Variable
    vector<Eigen::Vector2d> z(3);    // 观测 Measurements Vector
    z[0] = Eigen::Vector2d (0.37, 1.4);
    z[1] = Eigen::Vector2d (1.38, 2.6);
    z[2] = Eigen::Vector2d (2.77, 3.2);

    Eigen::Vector2d G; // Control Matrix
    G << 0.5*dt*dt,
         dt;

    Eigen::Matrix2d H; // Observation Matrix
    H << 1, 0,
         0, 1;

    Eigen::Matrix2d Q; //Process Noise Covariance
    Q << 0.05, 0,
         0, 0.05;

    Eigen::Matrix2d R; //Measurement Covariance
    R << 0.3, 0,
         0, 0.3;


    Eigen::Matrix2d P = Eigen::Matrix2d::Identity();

    for (int i = 0; i < 3; i++)
    {
        // 写出预测过程
        Eigen::Vector2d x_predicted = F * x + G * u[i];
        Eigen::Matrix2d P_predicted = F * P * F.transpose() + Q;

        // 写出更新过程
        Eigen::Matrix2d K = P_predicted * H.transpose() * (H * P_predicted * H.transpose() + R).inverse();
        Eigen::Vector2d x_updated = x_predicted + K * (z[i] - H * x_predicted);
        Eigen::Matrix2d P_updated = (Eigen::Matrix2d::Identity() - K * H) * P_predicted;

        x = x_updated;
        P = P_updated;

        std::cout << "predict state x: " << x_predicted.transpose() << endl;
        std::cout << "Measurement state x: " << z[i].transpose() << endl;
        std::cout << "Update state x: " << x.transpose() << endl;
        std::cout << "covariance P: " << P << endl;
        std::cout<< "Kgain" << K <<std::endl;
        std::cout<<"\n"<<std::endl;
    }
    
    return 0;
}
