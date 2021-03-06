cmake_minimum_required(VERSION 3.12)
project(ceres_example)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(PCL REQUIRED)
list (REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(helloWorld hello_world.cpp)
target_link_libraries(helloWorld ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(powell powell.cpp)
target_link_libraries(powell ${CERES_LIBRARIES})

add_executable(curveFitting curve_fitting.cpp)
target_link_libraries(curveFitting ${CERES_LIBRARIES})

add_executable(robustCurve robust_curve.cpp)
target_link_libraries(robustCurve ${CERES_LIBRARIES})

add_executable(simpleBA simple_BA.cpp)
target_link_libraries(simpleBA ${CERES_LIBRARIES})

add_executable(StereoBA StereoBA.cpp)
target_link_libraries(StereoBA ${CERES_LIBRARIES})

#add_executable(linePNP line_pnp.cpp)
#target_link_libraries(linePNP ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(poseGraph poseGraph.cpp)
target_link_libraries(poseGraph ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(PoseGraph PoseGraph.cpp)
target_link_libraries(PoseGraph ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

add_executable(ICP ICP.cpp)
target_link_libraries(ICP ${CERES_LIBRARIES} ${Sophus_LIBRARIES} ${PCL_LIBRARIES})