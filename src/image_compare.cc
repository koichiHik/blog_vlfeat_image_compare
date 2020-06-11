

// STL
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
#include <cereal/types/vector.hpp>

// Eigen
#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Original
#include "eigen_serializable.h"
#include "image_descriptor.h"

DEFINE_string(image_directory, "", "");
DEFINE_string(data_directory, "", "");

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
                   std::unordered_map<std::string, ImageInfo>& iamge_info_map) {
  CHECK_GE(image_files.size(), feature_files.size());

  for (const auto feature_path : feature_files) {
    const std::string filename_wo_ext =
        std::filesystem::path(feature_path).filename().replace_extension("");
    std::string image_path = GetFilePath(filename_wo_ext, image_files);

    cv::Mat image = cv::imread(image_path);
    std::vector<KeyPoint> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    LoadKeyPointsAndFeatures(feature_path, keypoints, descriptors);
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

}  // namespace

int main(int argc, char** argv) {
  // X. Initial setting.
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  // X. Data directory.
  const std::string image_dir =
      FLAGS_image_directory.empty() ? project_folder_path + "image" : FLAGS_image_directory;
  const std::string data_dir =
      FLAGS_data_directory.empty() ? project_folder_path + "/data" : FLAGS_data_directory;

  // X. Extract binary file paths.
  std::vector<std::string> image_files, feature_files;
  LOG(INFO) << "Loag all image file paths.";
  ExtractAllFilePathsInDirectory(image_dir, image_files);
  LOG(INFO) << "Loag all binary file paths.";
  ExtractAllFilePathsInDirectory(data_dir, feature_files);

  // X. Deserialize
  std::unordered_map<std::string, ImageInfo> image_info_map;
  LoadImageInfo(image_files, feature_files, image_info_map);

  // X. Display Key Points.
  DisplayKeyPoints(image_info_map);

  return 0;
}
