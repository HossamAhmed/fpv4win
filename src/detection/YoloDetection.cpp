#include "YoloDetection.h"
#include <QDebug>

YoloDetection::YoloDetection(QObject *parent)
    : QObject{parent}
{}

void YoloDetection::startDetection()
{
    qDebug()<<"YoloDetection -> startDetection ....";
}
