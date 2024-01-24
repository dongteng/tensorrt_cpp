
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
        //����������ļ�һЩ����Severity::kWARNING
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
} logger;



int main(){
    // =================== 1 ����builder ==================
    Logger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

    // ����network kEXPLICIT_BATCH������ʽbatch���Ƽ�ʹ�ã�����tensor�а���batch���γ�ȡ�
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder ->createNetworkV2(flag);


    // ��������ṹ
    // bert ����ṹ   ����input_ids  attention_mask  token_type_ids

    // =================== 3 ���� onnx ������ ===================
    const char* onnxModelPath = "/home/guest/user/zhjm/cppprojects/model_repo/model.onnx";
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // ��ȡģ���ļ��������Ƿ���ڴ���
    parser->parseFromFile(onnxModelPath,static_cast<int32_t> (Logger ::Severity::kWARNING));
    //parseFromFile ��������ģ���ļ���·����һ����־������Ϊ�����������ģ��·��Ϊ onnxModelPath����־����Ϊ���漶�� (TRTLogger::Severity::kWARNING)��
    //��ʹ���� C++ �е� static_cast ������� TRTLogger::Severity::kWARNING ת��Ϊ int32_t ���͡�

    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    std::cout << "successfully parse the onnx model" << std::endl;


    //===================== ���ñ�Ҫ���� config=============================
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    //���ù����ռ��С
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,100*(1<<20));

    //�����԰뾫�ȹ���engine  ������torchģ���е�������32λ��������fp32
    //tensorrt�п���ֱ�ӽ�Ȩ������ΪFP16���������ٶȣ���Ҫ����ΪINT8������Ҫ��������У׼��
    // INT8�������Բ���YOLO�Ĵ��롣
    // ���ﲻ����ģ��������ԭ��
    config->setFlag(nvinfer1::BuilderFlag::kFP16);


    //===================== 4 ����profile ����ά������=============================
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



    //��֮ǰ�������Ż������ļ� ��ӵ�TensorRT����IBuilderConfig��
    config->addOptimizationProfile(profile);


    //ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // ʹ�����ַ�ʽ��ģ��ֱ������Ϊһ�� TensorRT ���� (ICudaEngine*)������Ҫ��һ�������л����衣����������ֱ�������ƶϡ�
    // ���ַ�ʽ�ʺ����ڱ����豸�Ͻ����ƶϣ������漰ģ�͵Ŀ��豸�����洢��

    //���л�����engine ʹ�����ַ�ʽ��ģ�ͱ����л�Ϊһ���ֽ����У����洢�� IHostMemory* �����С�����Խ�����ֽ����б��浽�����ϵ��ļ��У�����ͨ�����紫�䵽��һ̨�豸��
    //���ַ�ʽ�ʺ�����Ҫ�ڲ�ͬ�豸�ϼ�����ͬģ�͵ĳ���������һ̨�豸�ϣ������ʹ�� TensorRT API ���´������档
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
//    assert(serializedModel != nullptr);

    const char* engine_filePath = "/home/guest/user/zhjm/cppprojects/model_repo/model_demo.engine";
    std::ofstream p(engine_filePath, std::ios::binary);
    //���д���ʹ���� C++ ���ļ������ (std::ofstream)�������Զ�����ģʽ (std::ios::binary) ��һ���ļ����ļ�·���� engine_filePath��
    //������˵����δ���������Ǵ���һ���ļ���������� p�����ڽ�����д�뵽ָ��·�����ļ��С�engine_filePath ���ļ���·������ std::ios::binary ��ʾ�Զ�����ģʽ���ļ���

    if (!p.is_open())
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }


    //reinterpret_cast<const char*>(serializedModel->data())���ⲿ�ִ���ʹ���� reinterpret_cast���� serializedModel->data() ��ָ��ת��Ϊ const char* ���͡�������Ϊ write �������� const char* ���͵����ݡ�
    //serializedModel->size()���ⲿ�ִ����ȡ�� serializedModel �д洢���ݵĴ�С�����ֽ�����
    //p.write(...)���ⲿ�ִ���������ļ���������� p �� write ��������ָ����С������д���ļ���
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