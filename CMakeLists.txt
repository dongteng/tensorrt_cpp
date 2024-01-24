cmake_minimum_required(VERSION 3.26)
project(m3e_tensorrt)

set(CMAKE_CXX_STANDARD 17)

# ���� TensorRT ��װ·��
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

# �ҵ� CUDA
find_package(CUDA REQUIRED)

# ���� CUDA Դ�ļ�
set(CUDA_SOURCES
#        build.cpp
        runtime.cu
)

# ���� CUDA ��ִ���ļ�
cuda_add_executable(m3e_tensorrt ${CUDA_SOURCES})

# ���� OpenCV ·�����ã����ʹ�� OpenCV��
#set(OpenCV_DIR /home/guest/user/zhjm/cppprojects/opencv-4.8.0/)
#list(APPEND CMAKE_PREFIX_PATH ${OpenCV_DIR})
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

# ���� TensorRT �� CUDA ·������
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(/home/guest/user/zhjm/cppprojects/TensorRT-8.6.1.6/sample/common/)

# ������һ��ָ�� CUBLAS ���·���������Ҫ��
link_directories(/usr/lib/x86_64-linux-gnu)
# ���� lnvonnxparser���·��
link_directories( /home/guest/user/zhjm/cppprojects/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/)


# ͷ�ļ�
# target_include_directoried( )

# ���� OpenCV��TensorRT �� CUDA ��
target_link_libraries(m3e_tensorrt  ${CUDA_LIBRARIES} ${TensorRT_LIBRARY} nvonnxparser)