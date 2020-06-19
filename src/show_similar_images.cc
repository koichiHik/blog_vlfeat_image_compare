

// STL
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

// Glog
#include <glog/logging.h>

// GFlag
#include <gflags/gflags.h>

// Cereal
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>

// Eigen
#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Matplotlib-CPP
#include "matplotlib-cpp/matplotlibcpp.h"

// Original
#include "eigen_serializable.h"
#include "global_descriptor.h"
#include "image_descriptor.h"

DEFINE_string(image_directory, "", "");
DEFINE_string(sift_directory, "", "");
DEFINE_string(matching_matrix_path, "", "");
DEFINE_string(target_filename, "", "");

// Value defined in CMakeLists.txt file.
static const std::string project_folder_path = PRJ_FOLDER_PATH;

namespace {

void ExtractAllFilePathsInDirectory(const std::string& dir_path,
                                    std::vector<std::string>& filenames) {
  namespace fs = std::filesystem;
  for (const fs::directory_entry& file : fs::directory_iterator(dir_path)) {
    if (!file.is_directory()) {
      filenames.push_back(file.path());
    }
  }
  LOG(INFO) << "Count of found files : " << filenames.size();
}

std::string GetFilePath(const std::string& partial_name,
                        const std::vector<std::string>& filepaths) {
  for (const auto path : filepaths) {
    if (path.rfind(partial_name) != std::string::npos) {
      return path;
    }
  }
  LOG(FATAL);
  return "";
}

void LoadKeyPointsAndFeatures(const std::string& bin_file_path, std::vector<KeyPoint>& keypoints,
                              std::vector<Eigen::VectorXf>& descriptors) {
  keypoints.clear();
  descriptors.clear();

  std::ifstream feature_reader(bin_file_path, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive input_archive(feature_reader);

  input_archive(keypoints, descriptors);
}

void LoadImageInfo(const std::vector<std::string>& image_files,
                   const std::vector<std::string>& feature_files,
                   std::unordered_map<std::string, ImageInfo>& iamge_info_map,
                   bool load_images = false) {
  if (load_images) {
    CHECK_GE(image_files.size(), feature_files.size());
  }

  int total_size = feature_files.size();
  int cnt = 0;
  for (const auto feature_path : feature_files) {
    const std::string filename_wo_ext =
        std::filesystem::path(feature_path).filename().replace_extension("");

    LOG(INFO) << "Load image info... " << cnt << " / " << total_size;
    cnt++;

    std::vector<KeyPoint> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    LoadKeyPointsAndFeatures(feature_path, keypoints, descriptors);

    cv::Mat image;
    if (load_images) {
      std::string image_path = GetFilePath(filename_wo_ext, image_files);
      image = cv::imread(image_path);
    }

    iamge_info_map[filename_wo_ext] = ImageInfo(image, keypoints, descriptors);
  }
}

void DrawKeyPoints(const std::vector<KeyPoint>& keypoints, cv::Mat& img) {
  for (const auto& kp : keypoints) {
    int radius = std::pow(2, kp.octave_);
    cv::circle(img, cv::Point2i(kp.x_, kp.y_), kp.size_, cv::Scalar(0, 200, 200), 1);
  }
}

void DisplayKeyPoints(const std::unordered_map<std::string, ImageInfo>& info_map) {
  for (const auto& [key, value] : info_map) {
    cv::Mat drawn_img;
    value.image_.copyTo(drawn_img);
    DrawKeyPoints(value.keypoints_, drawn_img);
    cv::imshow("Image", drawn_img);
    cv::waitKey(0);
  }
}

void CreateMatchingMatrix(const std::string& gmm_file_path, const std::string& matching_matrix_path,
                          std::unordered_map<std::string, ImageInfo>& image_info_map) {
  FisherVectorEncoder fisher_encoder(gmm_file_path);

  // X.
  LOG(INFO) << "Create filename index map.";
  std::unordered_map<std::string, int> filenames_indices;
  std::unordered_map<int, std::string> indices_filemanes;
  int index = 0;
  for (auto citr = image_info_map.cbegin(); citr != image_info_map.cend(); ++citr) {
    std::filesystem::path p(citr->first);
    std::string filename = p.replace_extension("").filename().string();
    filenames_indices[filename] = index;
    indices_filemanes[index] = filename;
    index++;
  }

  LOG(INFO) << "Matching matrix computation starts!";
  int size = image_info_map.size();
  Eigen::MatrixXf matching_matrix(size, size);

  LOG(INFO) << "Size : " << size;
  for (const auto& [filename1, value] : image_info_map) {
    int idx1 = filenames_indices[filename1];
    matching_matrix(idx1, idx1) = 0.0;

    for (int idx2 = idx1 + 1; idx2 < size; idx2++) {
      Eigen::VectorXf feature1 =
          fisher_encoder.ComputeFisherVector(image_info_map[filename1].descriptors_);

      std::string filename2 = indices_filemanes[idx2];
      Eigen::VectorXf feature2 =
          fisher_encoder.ComputeFisherVector(image_info_map[filename2].descriptors_);

      float score = (feature1 - feature2).squaredNorm();

      matching_matrix(idx1, idx2) = score;
      matching_matrix(idx2, idx1) = score;
    }
  }

  std::cout << matching_matrix << std::endl;

  std::ofstream writer(matching_matrix_path, std::ios::out | std::ios::binary);
  cereal::PortableBinaryOutputArchive output_archive(writer);
  output_archive(filenames_indices, matching_matrix);
}

void LoadMatchingMatrix(const std::string& filepath,
                        std::unordered_map<std::string, int>& filenames_indices,
                        Eigen::MatrixXf& matching_matrix) {
  std::ifstream reader(filepath, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive input_archive(reader);
  input_archive(filenames_indices, matching_matrix);
}

void ShowTopXImages(const std::string& target_filename, const Eigen::MatrixXf& matching_matrix,
                    int top_x, std::unordered_map<std::string, int>& filenames_indices,
                    std::unordered_map<std::string, ImageInfo>& image_info_map) {
  std::filesystem::path p(target_filename);
  std::string tgt = p.replace_extension("").filename();

  if (image_info_map.find(tgt) == image_info_map.end()) {
    LOG(INFO) << "File with name : " << tgt << " does not exist.";
  }

  // X. Create index - filename map.
  std::unordered_map<int, std::string> indices_filenames;
  for (const auto& [key, value] : filenames_indices) {
    indices_filenames[value] = key;
  }

  // X. Create Vector with Matching Score.
  int idx = filenames_indices[tgt];
  int len = matching_matrix.cols();
  std::vector<float> vec(len);
  for (int i = 0; i < len; i++) {
    vec[i] = matching_matrix(i, idx);
  }
  std::vector<float> sorted_score_vec(vec);
  std::sort(sorted_score_vec.begin(), sorted_score_vec.end());

  cv::imshow("Target Image", image_info_map[tgt].image_);

  for (int i = 0; i < top_x; i++) {
    int min_idx = std::min_element(vec.begin(), vec.end()) - vec.begin();
    float score = vec[min_idx];
    vec[min_idx] = std::numeric_limits<float>::max();

    // X. Skip query image.
    if (score == 0) {
      continue;
    }

    std::string filename = indices_filenames[min_idx];
    LOG(INFO) << "Image of score    : " << score;
    LOG(INFO) << "Filename of image : " << filename;
    cv::imshow("Similar Image", image_info_map[filename].image_);
    cv::waitKey(0);
  }

  // X. Skip query image.
  std::vector<float> plot_data(sorted_score_vec.begin() + 1, sorted_score_vec.end());
  matplotlibcpp::plot(plot_data);
  matplotlibcpp::show();
}

}  // namespace

int main(int argc, char** argv) {
  // X. Initial setting.
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  // X. Data directory.
  const std::string image_dir =
      FLAGS_image_directory.empty() ? project_folder_path + "/images" : FLAGS_image_directory;
  const std::string sift_dir =
      FLAGS_sift_directory.empty() ? project_folder_path + "/data" : FLAGS_sift_directory;

  // X. Extract binary file paths.
  std::vector<std::string> image_files, feature_files;
  LOG(INFO) << "Load all image file paths.";
  ExtractAllFilePathsInDirectory(image_dir, image_files);
  LOG(INFO) << "Load all binary file paths.";
  ExtractAllFilePathsInDirectory(sift_dir, feature_files);

  // X. Deserialize
  LOG(INFO) << "Load ImageInfo.";
  std::unordered_map<std::string, ImageInfo> image_info_map;
  LoadImageInfo(image_files, feature_files, image_info_map, true);

  // X. Train GMM with Deserialized data.
  LOG(INFO) << "Load matching matrix.";
  std::unordered_map<std::string, int> image_indices;
  Eigen::MatrixXf matching_matrix;
  LoadMatchingMatrix(FLAGS_matching_matrix_path, image_indices, matching_matrix);

  // X.
  ShowTopXImages(FLAGS_target_filename, matching_matrix, 20, image_indices, image_info_map);

  return 0;
}
