#include "YoloDetection.h"
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <algorithm>
#include <float.h>
#include <fstream>
#include <chrono>


// Constants
const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
const cv::Size2f MODEL_SHAPE(640, 640);
std::vector<std::string> class_list;

Ort::Env& get_ort_env() {
    qDebug() << "Singleton for ONNX Runtime environment...";
    try {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        return env;
    } catch (const Ort::Exception& e) {
        qDebug() << "ONNX Runtime error: " << e.what();
    } catch (const std::exception& e) {
        qDebug() << "Error: " << e.what();
    }

    // FATAL fallback — should never be reached, but prevents undefined behavior
    throw std::runtime_error("Failed to initialize ONNX Runtime environment.");
}

// Load class names from file
std::vector<std::string> load_class_list() {
std::vector<std::string> class_list;
std::ifstream ifs("coco-classes.txt");
if (!ifs.is_open()) {
throw std::runtime_error("Failed to open class list file");
}

std::string line;
while (getline(ifs, line)) {
class_list.push_back(line);
}
return class_list;
}

// Create ONNX Runtime session with proper CUDA support
Ort::Session create_onnx_session(const wchar_t* model_path, bool use_cuda) {
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(1);
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

if (use_cuda) {
// Use the correct CUDA provider API
OrtCUDAProviderOptions cuda_options;
cuda_options.device_id = 0;
session_options.AppendExecutionProvider_CUDA(cuda_options);
}

return Ort::Session(get_ort_env(), model_path, session_options);
}

// Main detection function
void detect(cv::Mat& image, Ort::Session& session) {
// Preprocess image
cv::Mat blob;
cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, MODEL_SHAPE, cv::Scalar(), true, false);

// Create input tensor
std::array<int64_t, 4> input_shape = { 1, 3, MODEL_SHAPE.height, MODEL_SHAPE.width };
Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
OrtAllocatorType::OrtArenaAllocator,
OrtMemType::OrtMemTypeDefault);

Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
memory_info,
blob.ptr<float>(),
blob.total(),
input_shape.data(),
input_shape.size());

// Run inference
const char* input_names[] = { "images" };
const char* output_names[] = { "output0" };

auto output_tensors = session.Run(
Ort::RunOptions{ nullptr },
input_names,
&input_tensor,
1,
output_names,
1);

// Process outputs
float* raw_data = output_tensors[0].GetTensorMutableData<float>();
auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, 85, N]

int dimensions = static_cast<int>(output_shape[1]);  // 85
int rows = static_cast<int>(output_shape[2]);        // N

// Transpose manually from [1, 85, N] to [N, 85]
std::vector<std::vector<float>> transposed(rows, std::vector<float>(dimensions));
for (int i = 0; i < dimensions; ++i) {
for (int j = 0; j < rows; ++j) {
transposed[j][i] = raw_data[i * rows + j];
}
}

std::vector<int> class_ids;
std::vector<float> confidences;
std::vector<cv::Rect> boxes;

for (int i = 0; i < rows; ++i) {
float* data = transposed[i].data();
float* classes_scores = data + 4;

cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
cv::Point class_id;
double max_class_score;
minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

if (max_class_score > SCORE_THRESHOLD) {
confidences.push_back(max_class_score);
class_ids.push_back(class_id.x);

float x = data[0];
float y = data[1];
float w = data[2];
float h = data[3];

int left = static_cast<int>(x * image.cols - w * image.cols / 2);
int top = static_cast<int>(y * image.rows - h * image.rows / 2);
int width = static_cast<int>(w * image.cols);
int height = static_cast<int>(h * image.rows);

boxes.push_back(cv::Rect(left, top, width, height));
}
}

// Apply NMS
std::vector<int> nms_result;
cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

// Draw results
for (unsigned long i = 0; i < nms_result.size(); ++i) {
int idx = nms_result[i];
cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 255), 2);
cv::putText(image, std::to_string(int(confidences[idx] * 100)) + "% " + class_list[class_ids[idx]],
cv::Point(boxes[idx].x, boxes[idx].y), 1, 3, cv::Scalar(0, 255, 255), 2);
}
}


YoloDetection::YoloDetection(QObject *parent)
    : QObject{parent}
{}


void YoloDetection::startDetection()
{
    try {
        qDebug() << "Initializing YOLO detection...";
        auto& env = get_ort_env();
        qDebug() << "ONNX Runtime initialized" ;

        qDebug() << "load_class_list" ;

        class_list = load_class_list();

        cv::VideoCapture capture(1); // Use default camera
        if (!capture.isOpened()) {
            throw std::runtime_error("Failed to open video capture");
        }
        

        bool use_cuda = true; // Set to true for GPU acceleration
        Ort::Session session = create_onnx_session(L"yolov8n.onnx", use_cuda);
        qDebug() << "Model loaded successfully" ;

        
        cv::Mat image = cv::imread("input.jpg");
        if (image.empty()) {
            qDebug() << "Failed to load image!" ;
            return ;
        }

        detect(image, session);
        cv::imwrite("output.jpg", image);
        qDebug() << "Detection completed successfully!";

        auto start = std::chrono::steady_clock::now();
int frame_count = 0;
float fps = 0.0;

while (true) {
    cv::Mat frame;
    capture.read(frame);
    if (frame.empty()) break;

    detect(frame, session);

    // Calculate FPS
    frame_count++;
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    if (elapsed.count() >= 1.0f) {
        fps = frame_count / elapsed.count();
        frame_count = 0;
        start = end;
    }

    // Display FPS
    std::string fps_label = cv::format("FPS: %.2f", fps);
    cv::putText(frame, fps_label, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    cv::imshow("YOLOv8 Object Detection", frame);

    if (cv::waitKey(1) == 27) { // ESC to exit
        break;
    }
}

capture.release();
cv::destroyAllWindows();


    }
    catch (const Ort::Exception& e) {
        qCritical() << "ONNX Runtime error:" << e.what();
    }
    catch (const std::exception& e) {
        qCritical() << "Error:" << e.what();
    }
}