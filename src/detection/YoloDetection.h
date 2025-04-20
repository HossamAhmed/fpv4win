#pragma once
#include <QObject>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "player/ffmpegInclude.h"
class YoloDetection : public QObject {
    Q_OBJECT
private:
    // Constants
    const std::vector<cv::Scalar> colors
        = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.5;
    const cv::Size2f MODEL_SHAPE { 640, 640 };
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int frame_count = 0;
    float fps = 0.0;
    Ort::Env &get_ort_env();
    std::vector<std::string> load_class_list();
    std::vector<std::string> class_list;
    Ort::Session session = create_onnx_session(L"yolov8n.onnx", true);
    Ort::Session create_onnx_session(const wchar_t *model_path, bool use_cuda);

public:
    explicit YoloDetection(QObject *parent = nullptr);
    cv::Mat AVFrameToMat(AVFrame *frame);
    void MatToAVFrame(const cv::Mat &mat, AVFrame *target_frame);
    void detect(cv::Mat &image);
    void init();

signals:
public slots:
    void startDetection();
};
