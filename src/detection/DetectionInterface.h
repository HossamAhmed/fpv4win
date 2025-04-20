#pragma once
#include <QObject>

class DetectionInterface : public QObject {
    Q_OBJECT

public:
    DetectionInterface(QObject *parent = nullptr);

public slots:
    void startDetection();
signals:
};
