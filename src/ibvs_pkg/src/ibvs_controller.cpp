#include "ibvs_controller.hpp"


IbvsController::Features IbvsController::pixelsToNormalizedPoints(const PixelPoints& pixel_points)const{
    Features features;
    const double fx = K_(0, 0);
    const double fy = K_(1, 1);
    const double cx = K_(0, 2);
    const double cy = K_(1, 2);
    for (std::size_t i = 0; i < pixel_points.size(); ++i) {
        const double u=static_cast<double>(pixel_points[i].x);

        const double v=static_cast<double>(pixel_points[i].y);

        features(static_cast<Eigen::Index>(i),0)=(u - cx) / fx;

        features(static_cast<Eigen::Index>(i),1)=(v - cy) / fy;
    }
    return features;
}

IbvsController::InteractionMatrix IbvsController::interactionMatrix(double x,double y, double depth)const{
    InteractionMatrix L;
    L << -1.0 / depth,
          0.0,
          x / depth,
          x * y,
          -(1.0 + x * x),
          y,

          0.0,
          -1.0 / depth,
          y / depth,
          1.0 + y * y,
          -x * y,
          -x;

    return L;
}

IbvsController::MultiInteractionMatrices IbvsController::multiInteractionMatrices(const Features& features,const Depths& depths)const{
    MultiInteractionMatrices L;
    for (Eigen::Index i = 0; i < 4; ++i) {
        const double x = features(i, 0);
        const double y = features(i, 1);
        const double depth = depths(i);
        L.block<2, 6>(2 * i, 0)=interactionMatrix(x,y,depth);
    }
    return L;
}

IbvsController::CameraVelocity IbvsController::computeCameraVelocity(const MultiInteractionMatrices& L,const FeatureError& error)const{
    Eigen::JacobiSVD<MultiInteractionMatrices> svd(
        L,
        Eigen::ComputeFullU |
        Eigen::ComputeFullV
    );
    svd.setThreshold(1e-6);
    CameraVelocity velocity=-gain_ * svd.solve(error);
    return velocity;
}

IbvsController::CameraVelocity IbvsController::limitVelocity(const CameraVelocity& velocity)const{
    CameraVelocity limited_velocity=velocity;

    Eigen::Vector3d linear_velocity=limited_velocity.head<3>();

    Eigen::Vector3d angular_velocity=limited_velocity.tail<3>();

    const double linear_norm=linear_velocity.norm();

    const double angular_norm=angular_velocity.norm();

    if (linear_norm > max_linear_) {
        linear_velocity *= max_linear_ / linear_norm;
    }

    if (angular_norm > max_angular_) {
        angular_velocity *= max_angular_ / angular_norm;
    }

    limited_velocity.head<3>()=linear_velocity;

    limited_velocity.tail<3>()=angular_velocity;

    return limited_velocity;
}

IbvsController::CameraVelocity IbvsController::computeCommand(
        const PixelPoints& current_pixel_points,
        const PixelPoints& desired_pixel_points,
        const Depths& current_depths,
        InteractionMode mode,
        const MultiInteractionMatrices* desired_L
    )const{
        const Features current_features=pixelsToNormalizedPoints(current_pixel_points);
        const Features desired_features=pixelsToNormalizedPoints(desired_pixel_points);
        FeatureError error;
        for (Eigen::Index i = 0; i < 4; ++i) {
            error(2 * i) =current_features(i, 0)-desired_features(i, 0);
            error(2 * i + 1) =current_features(i, 1)-desired_features(i, 1);
        }
        const double error_norm=error.norm();
        if (error_norm<tolerance_){
            return CameraVelocity::Zero();
        }
        const MultiInteractionMatrices current_L=multiInteractionMatrices(
            current_features,
            current_depths
        );

        MultiInteractionMatrices used_L;
        switch (mode)
        {
        case InteractionMode::Current:
            used_L=current_L;
            break;
        case InteractionMode::Desired:
            used_L=*desired_L;
            break;
        case InteractionMode::Average:
            used_L=0.5*(current_L+*desired_L);
            break;
        
        default:
            break;
        }
        const CameraVelocity raw_velocity=computeCameraVelocity(used_L,error);
        return limitVelocity(raw_velocity);
    }