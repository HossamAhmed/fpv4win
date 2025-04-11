#include "YoloDetection.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <algorithm>

YoloDetection::YoloDetection(QObject *parent)
    : QObject{parent}
{}






struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

std::vector<Detection> postprocess(
    const cv::Mat& input_image,
    const std::vector<Ort::Value>& output_tensors,
    float conf_threshold = 0.5,
    float iou_threshold = 0.5
) {
    std::vector<Detection> detections;
    
    // Get output tensor
    auto* raw_output = output_tensors[0].GetTensorData<float>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // Shape: [1, 84, 8400]
    
    // 8400 detections, 84 elements per detection (4 box + 80 class scores)
    const int num_classes = 80;
    const int num_anchors = output_shape[2];
    
    cv::Size resized_size = cv::Size(640, 640);
    float* data = (float*)raw_output;
    
    for (int i = 0; i < num_anchors; ++i) {
        float* detection = &data[i * output_shape[1]];
        
        // Get class scores
        float* classes_scores = detection + 4;
        int class_id = std::max_element(classes_scores, classes_scores + num_classes) - classes_scores;
        float confidence = classes_scores[class_id];
        
        if (confidence > conf_threshold) {
            // Get box coordinates (center x, center y, width, height)
            float cx = detection[0];
            float cy = detection[1];
            float w = detection[2];
            float h = detection[3];
            
            // Convert to x1,y1,x2,y2
            float x1 = (cx - w/2) / resized_size.width * input_image.cols;
            float y1 = (cy - h/2) / resized_size.height * input_image.rows;
            float x2 = (cx + w/2) / resized_size.width * input_image.cols;
            float y2 = (cy + h/2) / resized_size.height * input_image.rows;
            
            detections.push_back(Detection{
                cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                confidence,
                class_id
            });
        }
    }
    
    // Basic NMS implementation
    std::vector<bool> keep(detections.size(), true);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j]) continue;
            cv::Rect a = detections[i].box;
            cv::Rect b = detections[j].box;
            float intersection_area = (a & b).area();
            float union_area = a.area() + b.area() - intersection_area;
            if (intersection_area / union_area > iou_threshold) {
                if (detections[j].conf > detections[i].conf)
                    keep[i] = false;
                else
                    keep[j] = false;
            }
        }
    }
    
    std::vector<Detection> final_detections;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (keep[i]) final_detections.push_back(detections[i]);
    }
    
    return final_detections;
}

void YoloDetection::startDetection()
{
    try{
    qDebug()<<"YoloDetection -> startDetection ....";

    qDebug()<<"Initialize ONNX Runtime ....";
    // Initialize ONNX Runtime
    qDebug()<<"Ort::Env ....";
  

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
 
    qDebug()<<"Ort::SessionOptions ....";
    Ort::SessionOptions session_options;
    qDebug()<<"session_options.SetIntraOpNumThreads(1); ....";
    session_options.SetIntraOpNumThreads(1);
    
    qDebug()<<"Load model ....";
    // Load model
    Ort::Session session(env, L"yolov8n.onnx", session_options);
    
    qDebug()<<"Load image ....";
    // Load image
    cv::Mat image = cv::imread("test.jpg");
    if(image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    
    qDebug()<<"Preprocess image ....";
    // Preprocess image
    cv::Mat blob;
    cv::dnn::blobFromImage(resized_image, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    
    qDebug()<<"Create input tensor ....";
    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        blob.ptr<float>(),
        blob.total(),
        input_shape.data(),
        input_shape.size()
    );
    
    qDebug()<<"Run inference ....";
    // Run inference
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};
    std::vector<Ort::Value> outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    
    qDebug()<<"Postprocess ....";
    // Postprocess
    std::vector<Detection> detections = postprocess(image, outputs);
    
    qDebug()<<"Draw results ....";
    // Draw results
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(image,
                   std::to_string(det.class_id) + ": " + std::to_string(det.conf),
                   det.box.tl(),
                   cv::FONT_HERSHEY_SIMPLEX,
                   0.5,
                   cv::Scalar(0, 255, 0),
                   1);
    }
    
    cv::imwrite("result.jpg", image);
    std::cout << "Detection completed successfully!" << std::endl;
}
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        
    }
    catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        
    }
    
}