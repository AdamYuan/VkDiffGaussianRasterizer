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

add_executable(VkDiffGaussianRasterizer
        src/main.cpp

        src/Rasterizer.hpp
        src/Rasterizer.cpp

        src/ForwardRasterizer.hpp
        src/ForwardRasterizer.cpp

        src/DeviceSorter.hpp
        src/DeviceSorter.cpp
        src/DeviceSorterShader.cpp
)
target_link_libraries(VkDiffGaussianRasterizer PRIVATE
        myvk::vulkan myvk::glfw myvk::imgui # myvk::rg
        # glm::glm
        shader
)
