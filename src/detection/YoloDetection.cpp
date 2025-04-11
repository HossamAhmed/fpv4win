#include "YoloDetection.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <algorithm>
#include <float.h>

YoloDetection::YoloDetection(QObject *parent)
    : QObject{parent}
{}

const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Detection {
    cv::Rect box;
    float conf;
    int class_id;
};

cv::Mat letterbox_image(const cv::Mat& src, int target_width, int target_height, 
                       float& scale, int& pad_left, int& pad_top) {
    int width = src.cols;
    int height = src.rows;

    scale = std::min(static_cast<float>(target_width)/width, 
                    static_cast<float>(target_height)/height);
    int new_width = width * scale;
    int new_height = height * scale;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_width, new_height));

    pad_left = (target_width - new_width) / 2;
    pad_top = (target_height - new_height) / 2;

    cv::Mat padded(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, new_width, new_height)));

    return padded;
}

float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    cv::Rect intersection = rect1 & rect2;
    float intersectionArea = intersection.area();
    float unionArea = rect1.area() + rect2.area() - intersectionArea;
    return intersectionArea / unionArea;
}

std::vector<Detection> postprocess(
    const cv::Mat& input_image,
    const std::vector<Ort::Value>& output_tensors,
    float scale,
    int pad_left,
    int pad_top,
    float conf_threshold = 0.25,
    float iou_threshold = 0.45
) {
    std::vector<Detection> detections;

    auto* data = output_tensors[0].GetTensorData<float>();
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_classes = 80;
    const int num_anchors = output_shape[2];

    for (int i = 0; i < num_anchors; ++i) {
        float cx = data[i];
        float cy = data[num_anchors + i];
        float w = data[2 * num_anchors + i];
        float h = data[3 * num_anchors + i];

        int class_id = -1;
        float max_score = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_threshold && class_id != -1) {
            float x1 = (cx - w/2 - pad_left) / scale;
            float y1 = (cy - h/2 - pad_top) / scale;
            float x2 = (cx + w/2 - pad_left) / scale;
            float y2 = (cy + h/2 - pad_top) / scale;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(input_image.cols)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(input_image.rows)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(input_image.cols)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(input_image.rows)));

            detections.emplace_back(
                cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                max_score,
                class_id
            );
        }
    }

    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<bool> keep(detections.size(), true);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j]) continue;
            float iou = calculateIOU(detections[i].box, detections[j].box);
            if (iou >= iou_threshold) {
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
    try {
        qDebug() << "Initializing YOLO detection...";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        qDebug() << "Loading ONNX model...";
        Ort::Session session(env, L"yolov8n.onnx", session_options);

        qDebug() << "Processing image...";
        cv::Mat image = cv::imread("test.jpg");
        if(image.empty()) {
            qWarning() << "Failed to load image!";
            return;
        }

        float scale;
        int pad_left, pad_top;
        cv::Mat processed_image = letterbox_image(image, 640, 640, scale, pad_left, pad_top);

        cv::Mat blob;
        cv::dnn::blobFromImage(processed_image, blob, 1/255.0, cv::Size(), cv::Scalar(), true, false);

        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob.ptr<float>(),
            blob.total(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};
        qDebug() << "Running inference...";
        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                 input_names,
                                 &input_tensor,
                                 1,
                                 output_names,
                                 1);

        qDebug() << "Postprocessing results...";
        auto detections = postprocess(image, outputs, scale, pad_left, pad_top);

        qDebug() << "Drawing detections...";
        for (const auto& det : detections) {
            qDebug() << QString::fromStdString(class_names[det.class_id]) ;
            qDebug().nospace() 
            << "Detected: " << QString::fromStdString(class_names[det.class_id]).leftJustified(15, ' ')
            << " Conf: " << QString::number(static_cast<double>(det.conf), 'f', 2)
            << " Box: [" 
            << "x:" << det.box.x << ", "
            << "y:" << det.box.y << ", "
            << "w:" << det.box.width << ", "
            << "h:" << det.box.height << "]";
    
            cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(image,
                       class_names[det.class_id] + ": " + std::to_string(det.conf).substr(0,4),
                       det.box.tl() + cv::Point(0, -5),
                       cv::FONT_HERSHEY_SIMPLEX,
                       0.5,
                       cv::Scalar(0, 255, 0),
                       1);
        }

        cv::imwrite("result.jpg", image);
        qDebug() << "Detection completed successfully!";
    }
    catch (const Ort::Exception& e) {
        qCritical() << "ONNX Runtime error:" << e.what();
    }
    catch (const std::exception& e) {
        qCritical() << "Error:" << e.what();
    }
}