
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cassert>
#include <fstream>
typedef char AsciiChar;
using namespace nvonnxparser;


class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        //这里如果看的简单一些就用Severity::kWARNING
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;



int main(){
    // =================== 1 创建builder ==================
    Logger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

    // 创建network kEXPLICIT_BATCH代表显式batch（推荐使用），即tensor中包含batch这个纬度。
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder ->createNetworkV2(flag);


    // 定义网络结构
    // bert 网络结构   定义input_ids  attention_mask  token_type_ids

    // =================== 3 创建 onnx 解析器 ===================
    const char* onnxModelPath = "/home/guest/user/zhjm/cppprojects/model_repo/model.onnx";
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // 读取模型文件并处理是否存在错误
    parser->parseFromFile(onnxModelPath,static_cast<int32_t> (Logger ::Severity::kWARNING));
    //parseFromFile 函数接受模型文件的路径和一个日志级别作为参数。在这里，模型路径为 onnxModelPath，日志级别为警告级别 (TRTLogger::Severity::kWARNING)。
    //它使用了 C++ 中的 static_cast 运算符将 TRTLogger::Severity::kWARNING 转换为 int32_t 类型。

    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully parse the onnx model" << std::endl;


    //===================== 设置必要参数 config=============================
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    //设置工作空间大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,100*(1<<20));

    //设置以半精度构建engine  我们在torch模型中的数据是32位浮点数即fp32
    //tensorrt中可以直接将权重量化为FP16，以提升速度，若要量化为INT8，则需要设置数据校准。
    // INT8量化可以参照YOLO的代码。
    // 这里不介绍模型量化的原理。
    config->setFlag(nvinfer1::BuilderFlag::kFP16);


    //===================== 4 设置profile 进行维度设置=============================
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input_ids",nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims2{1,64});
    profile->setDimensions("input_ids",nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims2{6,64});
    profile->setDimensions("input_ids",nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims2{11,64});

    profile->setDimensions("attention_mask",nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims2{1,64});
    profile->setDimensions("attention_mask",nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims2{6,64});
    profile->setDimensions("attention_mask",nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims2{11,64});

    profile->setDimensions("token_type_ids",nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims2{1,64});
    profile->setDimensions("token_type_ids",nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims2{6,64});
    profile->setDimensions("token_type_ids",nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims2{11,64});



    //将之前创建的优化配置文件 添加到TensorRT配置IBuilderConfig中
    config->addOptimizationProfile(profile);


    //ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // 使用这种方式，模型直接生成为一个 TensorRT 引擎 (ICudaEngine*)，不需要进一步的序列化步骤。这个引擎可以直接用于推断。
    // 这种方式适合于在本地设备上进行推断，而不涉及模型的跨设备传输或存储。

    //序列化构建engine 使用这种方式，模型被序列化为一个字节序列，并存储在 IHostMemory* 对象中。你可以将这个字节序列保存到磁盘上的文件中，或者通过网络传输到另一台设备。
    //这种方式适合于需要在不同设备上加载相同模型的场景。在另一台设备上，你可以使用 TensorRT API 重新创建引擎。
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
//    assert(serializedModel != nullptr);

    const char* engine_filePath = "/home/guest/user/zhjm/cppprojects/model_repo/model_demo.engine";
    std::ofstream p(engine_filePath, std::ios::binary);
    //这行代码使用了 C++ 的文件输出流 (std::ofstream)，用于以二进制模式 (std::ios::binary) 打开一个文件，文件路径是 engine_filePath。
    //具体来说，这段代码的作用是创建一个文件输出流对象 p，用于将数据写入到指定路径的文件中。engine_filePath 是文件的路径，而 std::ios::binary 表示以二进制模式打开文件。

    if (!p.is_open())
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }


    //reinterpret_cast<const char*>(serializedModel->data())：这部分代码使用了 reinterpret_cast，将 serializedModel->data() 的指针转换为 const char* 类型。这是因为 write 函数接受 const char* 类型的数据。
    //serializedModel->size()：这部分代码获取了 serializedModel 中存储数据的大小，即字节数。
    //p.write(...)：这部分代码调用了文件输出流对象 p 的 write 方法，将指定大小的数据写入文件。
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    p.close();
    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedModel;

    std::cout << "===== work  done ====="<<std::endl;
    return 0;
}