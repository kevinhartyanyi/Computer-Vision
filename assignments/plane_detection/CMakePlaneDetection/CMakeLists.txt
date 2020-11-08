cmake_minimum_required(VERSION 3.0.0)
project(PlaneDetectionCMake VERSION 0.1.0)

INCLUDE_DIRECTORIES(
/usr/share/vcpkg/installed/x64-linux/include
)
LINK_DIRECTORIES(
   /usr/share/vcpkg/installed/x64-linux/lib
) 

add_executable(PlaneDetectionCMake main.cpp)

TARGET_LINK_LIBRARIES(PlaneDetectionCMake
    opencv3
)
