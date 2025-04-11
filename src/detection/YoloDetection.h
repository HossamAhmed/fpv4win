#pragma once


#include <QObject>

class YoloDetection : public QObject
{
    Q_OBJECT
public:
    explicit YoloDetection(QObject *parent = nullptr);

signals:
public slots:
    void startDetection();
};

