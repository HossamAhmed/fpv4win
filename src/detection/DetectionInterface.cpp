#include "DetectionInterface.h"
#include "YoloDetection.h"
#include <QDebug>

DetectionInterface::DetectionInterface(QObject *parent)
    : QObject { parent } {

    qDebug() << "DetectionInterface constructor called";
}

void DetectionInterface::startDetection() {
    qDebug() << "Start Detection from DetectionInterface";
    YoloDetection detection;
    detection.startDetection();
}
