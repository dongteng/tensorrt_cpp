#include<iostream>
#include<vector>
#include <fstream>
#include <cassert>
#include"NvInfer.h"



class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        //����������ļ�һЩ����Severity::kWARNING
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;
//����ģ��
std::vector<unsigned char> loadEngineModel(const std::string &fileName)
{
    std::ifstream file(fileName, std::ios::binary);        // �Զ����Ʒ�ʽ��ȡ
    assert(file.is_open() && "load engine model failed!"); // ����

    file.seekg(0, std::ios::end); // ��λ���ļ�ĩβ
    size_t size = file.tellg();   // ��ȡ�ļ���С

    std::vector<unsigned char> data(size); // ����һ��vector����СΪsize
    file.seekg(0, std::ios::beg);          // ��λ���ļ���ͷ
    file.read((char *)data.data(), size);  // ��ȡ�ļ����ݵ�data��
    file.close();

    return data;
}

int main() {
    // =========== 1 ����һ����������ʱruntime ===========
    Logger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);

    // =========== 2 �����л�����engine ============
    // ��ȡ�ļ�
    auto engineModel = loadEngineModel("/home/guest/user/zhjm/cppprojects/model_repo/model_demo.engine");
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr);

    if (!engine) {
        std::cout << "deserialize engine failed!" << std::endl;
        return -1;
    }

    // =========== 3 ����һ��ִ�������� ============
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "context create failed " << std::endl;
        return -1;
    }

    // ����������������״
    const int batch_size = 1; // ����ʵ���������
    const int sequence_length = 64; // ����ʵ���������
    const int input_size = batch_size * sequence_length;


    //����������״
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
    // =========== 5 ����͸�������������ݵ��豸 ============
    // ��������� device �ڴ�
    void* gpuInputIds, *gpuTokenTypeId, *gpuAttentionMask, *gpuOutput0,*gpuOutput1;
    cudaMalloc(&gpuInputIds, batch_size * sequence_length * sizeof(int64_t));
    cudaMalloc(&gpuTokenTypeId, batch_size * sequence_length  * sizeof(int64_t));
    cudaMalloc(&gpuAttentionMask, batch_size * sequence_length  * sizeof(int64_t));
    cudaMalloc(&gpuOutput0, batch_size * sequence_length * 1024* sizeof(float));
    cudaMalloc(&gpuOutput1, batch_size * 1024* sizeof(float));

    // �����ݴ��������Ƶ��豸
    cudaMemcpyAsync(gpuInputIds, inputIds.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpuTokenTypeId, tokenTypeIds.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(gpuAttentionMask, attentionMask.data(), batch_size * sequence_length * sizeof(int64_t), cudaMemcpyHostToDevice);


    // ===========  6 ִ������ ============
    void* bindings[] = {gpuInputIds, gpuTokenTypeId, gpuAttentionMask, gpuOutput0, gpuOutput1};
    bool success = context->enqueueV2( bindings, cudaStreamDefault, nullptr);
    if (!success) {
        std::cerr << "enqueueV2 failed!" << std::endl;
        return -1;
    }

    // =========== 7 ����������ݵ����� ============
    std::vector<float> output0(1024);
    std::vector<float> output1(1024);
//    cudaMemcpy(output0.data(), gpuOutput0, input_size * sizeof(float), cudaMemcpyDeviceToHost); ��Ϊ����Ҫ���ֵ ���Դ���Դ�����ν

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
    // ===========  8 �ͷ���Դ  ============
    cudaFree(gpuInputIds);
    cudaFree(gpuTokenTypeId);
    cudaFree(gpuAttentionMask);
    cudaFree(gpuOutput1);
    cudaFree(gpuOutput0);
    // ����TensorRT����
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
