
#ifndef _GLOBAL_DESCRIPTOR_H_
#define _GLOBAL_DESCRIPTOR_H_

// STL
#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Vlfeat
#include <vl/gmm.h>

// Eigen
#include <eigen3/Eigen/Core>

// Original
#include "image_descriptor.h"

class FisherVectorEncoder {
 public:
  FisherVectorEncoder(uint64_t num_dimension, uint64_t num_clusters);

  void TrainGMM(uint64_t max_itr, const std::vector<Eigen::VectorXf>& train_data);

  void TrainGMM(uint64_t max_itr, const std::unordered_map<std::string, ImageInfo>& image_info_map);

  Eigen::VectorXf ComputeFisherVector(const std::vector<Eigen::VectorXf>& features);

  void SaveGMM(const std::string& filepath);

  void LoadGMM(const std::string& filepath);

 private:
  std::vector<float> means_, covariances_, priors_, posteriors_;
  int num_dimension_;
  int num_clusters_;
  int num_data_;
};

#endif