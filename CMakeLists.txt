cmake_minimum_required(VERSION 2.8)
project(Lab6CV)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(ObjectTracker
	include/ObjectTracker.h
	src/ObjectTracker.cpp
)

add_executable(${PROJECT_NAME}
	src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})