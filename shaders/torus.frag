#version 450
layout(location = 0) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 tint;
} pc;

void main() {
    outColor = vec4(vertexColor * pc.tint.rgb, pc.tint.a);
}
