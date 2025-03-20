cmake_minimum_required(VERSION 3.15)
project(VkDiffGaussianRasterizer)

find_program(GLSLC "glslc" REQUIRED)

add_library(shader INTERFACE)

function(add_shader_library LIBRARY)
    add_library(shader_${LIBRARY} INTERFACE)
    add_library(shader::${LIBRARY} ALIAS shader_${LIBRARY})
    target_include_directories(shader_${LIBRARY} INTERFACE include)
    target_link_libraries(shader INTERFACE shader_${LIBRARY})
endfunction()

# Generated with DeepSeek
function(file_to_target_name FILE OUTPUT_VAR)
    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" SAFE_NAME "${FILE}")
    string(REGEX REPLACE "_+" "_" SAFE_NAME "${SAFE_NAME}")
    string(REGEX REPLACE "^_|_$" "" SAFE_NAME "${SAFE_NAME}")
    set(${OUTPUT_VAR} "${SAFE_NAME}" PARENT_SCOPE)
endfunction()

function(add_shader LIBRARY INPUT_FILE OUTPUT_FILE OPTIONS)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE}
            COMMAND ${GLSLC}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE}
            -o ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE}
            -mfmt=num
            --target-env=vulkan1.3
            ${OPTIONS}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE}
    )
    file_to_target_name(${OUTPUT_FILE} TARGET)
    add_custom_target(shader_${TARGET} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE})
    add_dependencies(shader_${LIBRARY} shader_${TARGET})
endfunction()

function(export_shader LIBRARY INPUT_FILE OUTPUT_FILE)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${INPUT_FILE}
    )
    file_to_target_name(${OUTPUT_FILE} TARGET)
    add_custom_target(shader_${TARGET} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${OUTPUT_FILE})
    add_dependencies(shader_${LIBRARY} shader_${TARGET})
endfunction()

function(add_shader_include SHADER_OUTPUT_FILE HEADER_FILE)
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/shader/${SHADER_OUTPUT_FILE}
            APPEND DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER_FILE}
    )
endfunction()

# Device Sorter
add_shader_library(DeviceSorter)
export_shader(DeviceSorter DeviceSorter/DeviceSorterSize.glsl DeviceSorter/DeviceSorterSize.hpp)
function(add_DeviceSorter_shader INPUT_FILE OUTPUT_FILE OPTIONS)
    add_shader(DeviceSorter DeviceSorter/${INPUT_FILE} DeviceSorter/${OUTPUT_FILE} "${OPTIONS}")
    add_shader_include(DeviceSorter/${OUTPUT_FILE} DeviceSorter/DeviceSorter.glsl)
    add_shader_include(DeviceSorter/${OUTPUT_FILE} DeviceSorter/DeviceSorterSize.glsl)
    add_shader_include(DeviceSorter/${OUTPUT_FILE} DeviceSorter/DeviceSorterWave.glsl)
endfunction()
add_DeviceSorter_shader(Reset.comp Reset.comp.inl "-O")
add_DeviceSorter_shader(GlobalHist.comp GlobalHist.comp.inl "-O")
add_DeviceSorter_shader(ScanHist.comp ScanHist32.comp.inl "-O;-DLANES_PER_WAVE=32")
add_DeviceSorter_shader(ScanHist.comp ScanHist64.comp.inl "-O;-DLANES_PER_WAVE=64")
add_DeviceSorter_shader(OneSweep.comp OneSweep32.comp.inl "-O;-DLANES_PER_WAVE=32")
add_DeviceSorter_shader(OneSweep.comp OneSweep64.comp.inl "-O;-DLANES_PER_WAVE=64")

# Device Sorter Test
add_shader_library(DeviceSorterTest)
add_shader(DeviceSorterTest DeviceSorterTest/Generate.comp DeviceSorterTest/Generate.comp.inl "-O")
add_shader(DeviceSorterTest DeviceSorterTest/Validate.comp DeviceSorterTest/Validate.comp.inl "-O")
export_shader(DeviceSorterTest DeviceSorterTest/Size.glsl DeviceSorterTest/Size.hpp)
