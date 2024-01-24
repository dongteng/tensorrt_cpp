#include<iostream>
#include<vector>
#include <fstream>
#include <cassert>
#include"NvInfer.h"



class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        //这里如果看的简单一些就用Severity::kWARNING
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;
//加载模型
std::vector<unsigned char> loadEngineModel(const std::string &fileName)
{
    std::ifstream file(fileName, std::ios::binary);        // 以二进制方式读取
    assert(file.is_open() && "load engine model failed!"); // 断言

    file.seekg(0, std::ios::end); // 定位到文件末尾
    size_t size = file.tellg();   // 获取文件大小

    std::vector<unsigned char> data(size); // 创建一个vector，大小为size
    file.seekg(0, std::ios::beg);          // 定位到文件开头
    file.read((char *)data.data(), size);  // 读取文件内容到data中
    file.close();

    return data;
}

int main() {
    // =========== 1 创建一个推理运行时runtime ===========
    Logger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);

    // =========== 2 反序列化生成engine ============
    // 读取文件
    auto engineModel = loadEngineModel("/home/guest/user/zhjm/cppprojects/model_repo/model_demo.engine");
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr);

    if (!engine) {
        std::cout << "deserialize engine failed!" << std::endl;
        return -1;
    }

    // =========== 3 创建一个执行上下文 ============
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "context create failed " << std::endl;
        return -1;
    }

    // 定义输入和输出的形状
    const int batch_size = 1; // 根据实际情况调整
    const int sequence_length = 64; // 根据实际情况调整
    const int input_size = batch_size * sequence_length;


    //设置输入形状
    for (int inputIndex = 0; inputIndex < engine->getNbBindings(); ++inputIndex)
    {
        if (engine->bindingIsInput(inputIndex)) {
            context->setBindingDimensions(inputIndex, nvinfer1::Dims2(batch_size, sequence_length));
        }
    }
    std::vector<int64_t> inputIds = {101, 791, 1921, 1921, 3698, 2582, 720, 3416, 102, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0};
    std::vector<int64_t> tokenTypeIds = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> attentionMask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // =========== 5 分配和复制输入输出数据到设备 ============
    // 分配输入的 device 内存
    void* gpuInputIds, *gpuTokenTypeId, *gpuAttentionMask, *gpuOutput0,*gpuOutput1;
    cudaMalloc(&gpuInputIds, batch_size * sequence_length * sizeof(int64_t));
    cudaMalloc(&gpuTokenTypeId, batch_size * sequence_length  * sizeof(int64_t));
    cudaMalloc(&gpuAttentionMask, batch_size * sequence_length  * sizeof(int64_t));
    cudaMalloc(&gpuOutput0, batch_size * sequence_length * 1024* sizeof(float));
    cudaMalloc(&gpuOutput1, batch_size * 1024* sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpyAsync(gpuInputIds, inputIds.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpuTokenTypeId, tokenTypeIds.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpuAttentionMask, attentionMask.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);


    // ===========  6 执行推理 ============
    void* bindings[] = {gpuInputIds, gpuTokenTypeId, gpuAttentionMask, gpuOutput0, gpuOutput1};
    bool success = context->enqueueV2( bindings, cudaStreamDefault, nullptr);
    if (!success) {
        std::cerr << "enqueueV2 failed!" << std::endl;
        return -1;
    }

    // =========== 7 复制输出数据到主机 ============
    std::vector<float> output0(1024);
    std::vector<float> output1(1024);
//    cudaMemcpy(output0.data(), gpuOutput0, input_size * sizeof(float), cudaMemcpyDeviceToHost); 因为不需要这个值 所以传输对错无所谓

    cudaMemcpy(output1.data(), gpuOutput1, 1024 * sizeof(float), cudaMemcpyDeviceToHost);


    int i = 0;
    int a = 0;
    for (float val : output1) {
        i = i + 1;
        std::cout << val << ", ";

        if (val != 0.0) {
            a = a + 1;
        }
    }
    std::cout << "\nthe total mount of output " << i << std::endl;
    std::cout << "\n the  mount not 0 " << a << std::endl;
    // ===========  8 释放资源  ============
    cudaFree(gpuInputIds);
    cudaFree(gpuTokenTypeId);
    cudaFree(gpuAttentionMask);
    cudaFree(gpuOutput1);
    cudaFree(gpuOutput0);
    // 销毁TensorRT对象
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
