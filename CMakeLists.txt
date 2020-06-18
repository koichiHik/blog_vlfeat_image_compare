
cmake_minimum_required(VERSION 3.8)
project(blog_vlfeat_sift)


# CXX Version.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/${ADDITIONAL_CMAKE_MODULE_DIR}")

FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Glog)
FIND_PACKAGE(Gflags)
FIND_PACKAGE(Eigen3)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/extern/lib/include)
link_directories(${CMAKE_SOURCE_DIR}/extern/lib)

############# Train GMM Model. #################
set(TARGET "train_gmm_model")
set(SRC src/train_gmm_model.cc src/global_descriptor.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})
target_link_libraries(${TARGET} vl)
target_link_libraries(${TARGET} stdc++fs)

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")

############# Create Image Matching Matrix. #################
set(TARGET "create_matching_matrix")
set(SRC src/create_matching_matrix.cc src/global_descriptor.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})
target_link_libraries(${TARGET} vl)
target_link_libraries(${TARGET} stdc++fs)

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")

############# Create Image Matching Matrix. #################
set(TARGET "show_similar_images")
set(SRC src/show_similar_images.cc src/global_descriptor.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})
target_link_libraries(${TARGET} vl)
target_link_libraries(${TARGET} stdc++fs)

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")