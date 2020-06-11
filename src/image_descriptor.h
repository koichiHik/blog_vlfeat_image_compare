
#ifndef _IMAGE_DESCRIPTOR_H_
#define _IMAGE_DESCRIPTOR_H_

// STL
#include <string>
#include <vector>

// Eigen
#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>

// Original
#include "keypoints.h"

class ImageInfo {
 public:
  ImageInfo() {}
  ImageInfo(cv::Mat& image, std::vector<KeyPoint>& keypoints,
            std::vector<Eigen::VectorXf>& descriptors)
      : image_(image), keypoints_(keypoints), descriptors_(descriptors) {}

  cv::Mat image_;
  std::vector<KeyPoint> keypoints_;
  std::vector<Eigen::VectorXf> descriptors_;
};

#endif