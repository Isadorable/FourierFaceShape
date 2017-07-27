#-------------------------------------------------
#
# Project created by QtCreator 2017-07-19T23:34:28
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = finaltest
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    ../dlib-19.4/dlib/all/source.cpp \
    cmshapecontext.cpp

INCLUDEPATH += /home/isadora/Downloads/dlib-19.4
QMAKE_CXXFLAGS +=  -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT

HEADERS += \
        mainwindow.h \
    cmshapecontext.h

FORMS += \
        mainwindow.ui

INCLUDEPATH += /opt/intel/mkl/include
LIBS += -L/opt/intel/mkl/lib/intel64 \
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
        -L/opt/intel/lib/intel64 \
        -liomp5 -lpthread -dl -lm



INCLUDEPATH += "/usr/local/include"
#LIBS += `pkg-config --libs opencv`
LIBS += -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core


LIBS += -lpthread -lX11 -ljpeg -lpng -DDLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT
LIBS += -L/home/user/libs/dlib-18.8
