TEMPLATE = app
CONFIG += c++11 qt console

QT += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

QMAKE_CXXFLAGS = -msse2 -msse3 -msse4.1

RESOURCES += \
    images.qrc

SOURCES += \
    main.cpp

HEADERS += \
    popcnt.h
