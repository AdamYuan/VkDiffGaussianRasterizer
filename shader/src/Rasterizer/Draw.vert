#version 460

out bOut { layout(location = 0) uint instanceID; }
gOut;

void main() { gOut.instanceID = gl_InstanceIndex; }
