cmake_minimum_required(VERSION 3.10)

project(NeuralNet_2 VERSION 0.1)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_subdirectory(extern/gtest)

set(Headers
)

set(Sources
	"./NeuralNet_2/src/Container.cpp"
)

#configure_file(./libconf.h.in ./libconf.h)

add_executable(${PROJECT_NAME} ${Sources} ${Headers})

add_subdirectory(test)
