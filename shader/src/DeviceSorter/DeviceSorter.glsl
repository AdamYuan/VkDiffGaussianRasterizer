#ifndef DEVICE_SORTER_GLSL
#define DEVICE_SORTER_GLSL

#include "DeviceSorterSize.glsl"

// Key Types
#define SortKey uint
#define SortPayload uint

layout(binding = 0) uniform bKeyCount {
	uint _padding[KEY_COUNT_BUFFER_OFFSET];
	uint gKeyCount;
};

uint divCeil(uint x, uint y) { return (x + y - 1) / y; }
uint getSortPartCount() { return divCeil(gKeyCount, SORT_PART_SIZE); }
uint extractKeyRadix(SortKey key, uint radixShift) { return uint((key >> radixShift) & (RADIX - 1)); }

// Decoupled look-back flags
#define FLAG_NOT_READY 0 // Flag value inidicating neither inclusive sum, nor reduction of a partition tile is ready
#define FLAG_REDUCTION 1 // Flag value indicating reduction of a partition tile is ready
#define FLAG_INCLUSIVE 2 // Flag value indicating inclusive sum of a partition tile is ready
#define FLAG_MASK 3      // Mask used to retrieve flag values

#endif
