#ifndef DEVICE_SORTER_WAVE_GLSL
#define DEVICE_SORTER_WAVE_GLSL

#ifndef LANES_PER_WAVE
#define LANES_PER_WAVE 32
#endif

#if LANES_PER_WAVE < 16
#error WaveGetLaneCount() < 16 not supported
#endif

#if LANES_PER_WAVE != 32 && LANES_PER_WAVE != 64 && LANES_PER_WAVE != 128
#error Unsupported LANES_PER_WAVE
#endif

#define WAVE_MASK_SIZE (LANES_PER_WAVE / 32)

#endif
