#!/bin/bash

# Install directory is relative to this script.
TOP_DIR=$(cd $(dirname $0); pwd)
INSTALL_DIR="${TOP_DIR}/extern/lib"
BUILD_DIR="${TOP_DIR}/extern/build"

################################
# Configure Cereal Library.    #
################################
if [ ! -e  "${BUILD_DIR}/cereal"  ]; then

mkdir -p "${BUILD_DIR}/cereal"
cd "${BUILD_DIR}/cereal"
cmake \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -D JUST_INSTALL_CEREAL=ON \
  -D SKIP_PORTABILITY_TEST=ON \
  -D SKIP_PERFORMANCE_COMPARISON=ON \
  ../../cereal
make install -j32
cd ${TOP_DIR}

fi

################################
# Configure Vlfeat Library.    #
################################
if [ ! -e "${INSTALL_DIR}/include/vl" ]; then
  mkdir -p ${INSTALL_DIR}
  mkdir -p "${INSTALL_DIR}/include"
fi

cd extern/vlfeat
make
cd ../../

cp -r extern/vlfeat/bin/glnxa64/libvl.so ${INSTALL_DIR}
cp -r extern/vlfeat/vl "${INSTALL_DIR}/include"

################################
# Configure Main Repository.   #
################################
ADDITIONAL_CMAKE_MODULE_DIR=./cmake

if [ ! -e ./build ]; then
  mkdir build
fi
cd build

cmake \
  -D ADDITIONAL_CMAKE_MODULE_DIR=${ADDITIONAL_CMAKE_MODULE_DIR} \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  ../

make

cd ../