
// STL
#include <fstream>

// Vlfeat
#include <vl/fisher.h>

// Glog
#include <glog/logging.h>

// Cereal
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>

// Original
#include "global_descriptor.h"

namespace {

Eigen::MatrixXf ConvertVectorOfFeaturesToMatrix(const std::vector<Eigen::VectorXf>& features) {
  int num_feature = features.size();
  int num_dimension = features[0].size();
  Eigen::MatrixXf feature_table(features[0].size(), features.size());
  for (int i = 0; i < feature_table.cols(); i++) {
    feature_table.col(i) = features[i];
  }
  return feature_table;
}

std::vector<Eigen::VectorXf> ConvertImageInfoToFeatureVectors(
    const std::unordered_map<std::string, ImageInfo>& image_info_map) {
  std::vector<Eigen::VectorXf> all_descriptors;

  LOG(INFO) << "Converting Image Info to Feature Vectors";
  int total_vectors = 0;
  for (const auto& [key, value] : image_info_map) {
    total_vectors += value.descriptors_.size();
  }

  int cnt = 0;
  LOG(INFO) << "Total count of feature vectors : " << total_vectors;
  all_descriptors.clear();
  all_descriptors.reserve(total_vectors);
  for (const auto& [key, value] : image_info_map) {
    all_descriptors.insert(all_descriptors.end(), value.descriptors_.begin(),
                           value.descriptors_.end());
  }
  return all_descriptors;
}

}  // namespace

FisherVectorEncoder::FisherVectorEncoder(uint64_t num_dimension, uint64_t num_clusters)
    : num_dimension_(num_dimension), num_clusters_(num_clusters) {}

void FisherVectorEncoder::TrainGMM(uint64_t max_itr,
                                   const std::vector<Eigen::VectorXf>& train_data) {
  // X. Create GMM object.
  LOG(INFO) << "vl_gmm_new : ";
  VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, num_dimension_, num_clusters_);

  // X. Set the maximum number of EM iterations to 100.
  LOG(INFO) << "vl_gmm_set_max_num_iterations : ";
  vl_gmm_set_max_num_iterations(gmm, max_itr);

  // X. Set the initialization to random selection.
  LOG(INFO) << "vl_gmm_set_initialization : ";
  vl_gmm_set_initialization(gmm, VlGMMRand);

  // X. Cluster the data. (Train the GMM).
  num_data_ = train_data.size();
  LOG(INFO) << "Converting to MatrixXf : ";
  Eigen::MatrixXf mat_data = ConvertVectorOfFeaturesToMatrix(train_data);

  LOG(INFO) << "Data count for training is : " << num_data_;
  vl_gmm_cluster(gmm, mat_data.data(), num_data_);

  LOG(INFO) << "Training is Done! Now storing data!";
  LOG(INFO) << "means copied!";
  means_.resize(num_dimension_ * num_clusters_);
  std::memcpy(means_.data(), vl_gmm_get_means(gmm), sizeof(float) * num_dimension_ * num_clusters_);

  LOG(INFO) << "covariances copied!";
  covariances_.resize(num_dimension_ * num_clusters_);
  std::memcpy(covariances_.data(), vl_gmm_get_covariances(gmm),
              sizeof(float) * num_dimension_ * num_clusters_);

  LOG(INFO) << "priors copied!";
  priors_.resize(num_clusters_);
  std::memcpy(priors_.data(), vl_gmm_get_priors(gmm), sizeof(float) * num_clusters_);

  LOG(INFO) << "posteriors copied!";
  posteriors_.resize(num_clusters_ * num_data_);
  std::memcpy(posteriors_.data(), vl_gmm_get_posteriors(gmm),
              sizeof(float) * num_clusters_ * num_data_);

  LOG(INFO) << "Deleting gmm";
  vl_gmm_delete(gmm);
  LOG(INFO) << "gmm deleted.";
}

void FisherVectorEncoder::TrainGMM(
    uint64_t max_itr, const std::unordered_map<std::string, ImageInfo>& image_info_map) {
  TrainGMM(max_itr, ConvertImageInfoToFeatureVectors(image_info_map));
}

Eigen::VectorXf FisherVectorEncoder::ComputeFisherVector(
    const std::vector<Eigen::VectorXf>& features) {
  Eigen::VectorXf fisher_vector(2 * features.size() * num_clusters_);

  Eigen::MatrixXf feature_matrix = ConvertVectorOfFeaturesToMatrix(features);

  vl_fisher_encode(fisher_vector.data(), VL_TYPE_FLOAT, means_.data(), num_dimension_,
                   num_clusters_, covariances_.data(), priors_.data(), feature_matrix.data(),
                   feature_matrix.cols(), VL_FISHER_FLAG_IMPROVED);

  return fisher_vector;
}

void FisherVectorEncoder::SaveGMM(const std::string& filepath) {
  // means : sizeof(float) * dimension x num_clusters
  // covariances : sizeof(float) * dimension x num_clusters
  // priors : sizeof(float) * num_clusters
  // posteriors : sizeof(float) * num_clusters * num_data

  LOG(INFO) << "Saving binary to : " << filepath;
  std::ofstream writer(filepath, std::ios::out | std::ios::binary);
  cereal::PortableBinaryOutputArchive output_archive(writer);
  output_archive(num_dimension_, num_clusters_, num_data_);
  output_archive(means_, covariances_, priors_, posteriors_);
}

void FisherVectorEncoder::LoadGMM(const std::string& filepath) {
  LOG(INFO) << "Loading binary from : " << filepath;
  std::ofstream reader(filepath, std::ios::in | std::ios::binary);
  cereal::PortableBinaryOutputArchive input_archive(reader);
  input_archive(num_dimension_, num_clusters_, num_data_);
  input_archive(means_, covariances_, priors_, posteriors_);
}