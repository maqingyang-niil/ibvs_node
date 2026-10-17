#pragma once

#include "ibvs_controller.hpp"

struct DetectionResult {
    bool success{false};
    IbvsController::PixelPoints pixel_points{};
    IbvsController::Depths depths{IbvsController::Depths::Zero()};
    double confidence{0.0};
};
