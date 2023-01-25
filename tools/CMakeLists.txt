
cmake_minimum_required(VERSION 3.19)

find_package(Torch REQUIRED)
find_package(torchtrt REQUIRED)
# target_include_directories(${TORCH_INCLUDE_DIRS} ${torchtrt_INCLUDE_DIRS} )
add_executable(converter converter.cpp)
target_link_libraries(converter torch torchtrt)
