cmake_minimum_required(VERSION 3.26)
project(m3e_tensorrt)

set(CMAKE_CXX_STANDARD 17)

# 设置 TensorRT 安装路径
set(TensorRT_ROOT /home/guest/user/zhjm/cppprojects/TensorRT-8.6.1.6/)
find_library(TensorRT_LIBRARY
        NAMES nvinfer
        PATHS ${TensorRT_ROOT}/lib
        PATH_SUFFIXES lib
        NO_DEFAULT_PATH
)
find_path(TensorRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS ${TensorRT_ROOT}/include
        NO_DEFAULT_PATH
)

# 找到 CUDA
find_package(CUDA REQUIRED)

# 设置 CUDA 源文件
set(CUDA_SOURCES
#        build.cpp
        runtime.cu
)

# 编译 CUDA 可执行文件
cuda_add_executable(m3e_tensorrt ${CUDA_SOURCES})

# 添加 OpenCV 路径设置（如果使用 OpenCV）
#set(OpenCV_DIR /home/guest/user/zhjm/cppprojects/opencv-4.8.0/)
#list(APPEND CMAKE_PREFIX_PATH ${OpenCV_DIR})
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

# 添加 TensorRT 和 CUDA 路径设置
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(/home/guest/user/zhjm/cppprojects/TensorRT-8.6.1.6/sample/common/)

# 添加这一行指定 CUBLAS 库的路径（如果需要）
link_directories(/usr/lib/x86_64-linux-gnu)
# 添加 lnvonnxparser库的路径
link_directories( /home/guest/user/zhjm/cppprojects/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/)


# 头文件
# target_include_directoried( )

# 链接 OpenCV、TensorRT 和 CUDA 库
target_link_libraries(m3e_tensorrt  ${CUDA_LIBRARIES} ${TensorRT_LIBRARY} nvonnxparser)
