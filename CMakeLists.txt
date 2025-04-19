cmake_minimum_required(VERSION 3.30)

project(LowLightEnhance)

set(CMAKE_CXX_STANDARD 20)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_FILES)
aux_source_directory("${PROJECT_SOURCE_DIR}/include" INCLUDE_FILES)

set(OpenCV_DIR lib/opencv)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TORCH_INSTALL_PREFIX lib/libtorch/debug)
    set(Torch_DIR lib/libtorch/debug/share/cmake/Torch)
else()
    set(TORCH_INSTALL_PREFIX lib/libtorch/release)
    set(Torch_DIR lib/libtorch/release/share/cmake/Torch)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})

target_include_directories(${PROJECT_NAME} PUBLIC "include")

file(GLOB TORCH_DLLS ${TORCH_INSTALL_PREFIX}/lib/*.dll)

file(GLOB OpenCV_DLLS ${OpenCV_DIR}/x64/vc16/bin/*.dll)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)


add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OpenCV_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
)