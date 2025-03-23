//
// Created by adamyuan on 3/17/25.
//

#include "Rasterizer.hpp"

#include <shader/DeviceSorter/Size.hpp>
#include <shader/Rasterizer/Size.hpp>

namespace VkGSRaster {

static_assert(KEY_COUNT_BUFFER_OFFSET == offsetof(VkDrawIndirectCommand, instanceCount));

} // namespace VkGSRaster