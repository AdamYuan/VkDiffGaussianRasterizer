cmake_minimum_required(VERSION 3.15)
project(VkDiffGaussianRasterizer)

set(CMAKE_CXX_STANDARD 20)

# Windows' math include does not define constants by default.
# Set this definition so it does.
# Also set NOMINMAX so the min and max functions are not overwritten with macros.
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
ENDIF ()

# Hide the console window in visual studio projects - Release
if (MSVC)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (NOT MINGW)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT IPO_SUPPORTED)
        if (IPO_SUPPORTED)
            message("Enable IPO")
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif ()
    endif ()
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
    endif ()
endif ()

add_subdirectory(dep/MyVK)
# add_subdirectory(dep/glm)

add_subdirectory(shader)

add_library(vkgsraster_DeviceSorter STATIC
        src/DeviceSorter.hpp
        src/DeviceSorter.cpp
        src/DeviceSorterShader.cpp
)
add_library(vkgsraster::DeviceSorter ALIAS vkgsraster_DeviceSorter)
target_link_libraries(vkgsraster_DeviceSorter PUBLIC
        myvk::vulkan
        shader::DeviceSorter
)

add_library(vkgsraster_Rasterizer STATIC
        src/Rasterizer.hpp
        src/Rasterizer.cpp
        src/RasterizerShader.cpp
)
add_library(vkgsraster::Rasterizer ALIAS vkgsraster_Rasterizer)
target_link_libraries(vkgsraster_Rasterizer PUBLIC
        myvk::vulkan
        vkgsraster::DeviceSorter
        shader::Rasterizer
)

# Tests

add_subdirectory(test/dep/json)
add_subdirectory(test/dep/tinyfiledialogs)
add_subdirectory(test/dep/diff-gaussian-rasterization)
add_subdirectory(test/dep/stb_image_write)

add_executable(DeviceSorterTest
        test/DeviceSorterTest.cpp
)
target_link_libraries(DeviceSorterTest PRIVATE
        myvk::vulkan myvk::glfw myvk::imgui
        vkgsraster::DeviceSorter
        shader::DeviceSorterTest
)

add_executable(RasterizerTest
        test/GSModel.cpp
        test/RasterizerTest.cpp
)
target_link_libraries(RasterizerTest PRIVATE
        myvk::vulkan myvk::glfw myvk::imgui
        vkgsraster::Rasterizer
        tinyfiledialogs
)

enable_language(CUDA)
add_executable(VerboseTest
        test/GSModel.cpp
        test/GSDataset.cpp
        test/VkCuBuffer.cu
        test/VkCuBuffer.hpp
        test/CuTileRasterizer.cu
        test/CuTileRasterizer.hpp
        test/VerboseTest.cpp
        test/VerboseTest.cu
)
target_link_libraries(VerboseTest PRIVATE
        myvk::vulkan
        vkgsraster::Rasterizer
        CudaRasterizer
        nlohmann_json
        stb_image_write
)
