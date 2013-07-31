#-------------------------------------------------
#
# Project created by QtCreator 2013-07-18T13:34:30
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = VideoTester
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    tutorialfunctions.cpp \
    handcaptureutilities.cpp

INCLUDEPATH += \
"C:\OpenCV_243\MiniGWBinaries\install\include"

LIBS += \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_calib3d245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_contrib245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_core245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_features2d245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_flann245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_gpu245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_highgui245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_imgproc245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_legacy245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_ml245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_nonfree245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_objdetect245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_photo245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_ts245.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_video245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_stitching245.dll.a" \
"C:\OpenCV_243\MiniGWBinaries\install\lib\libopencv_videostab245.dll.a"

HEADERS += \
    handTracker.h \
    tutorialfunctions.h \
    handcaptureutilities.h \
    fileEvaluator.h \
    videoRecorder.h
