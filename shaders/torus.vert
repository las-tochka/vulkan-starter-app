#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 vertexColor;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 worldPosition;

layout(set = 0, binding = 0) uniform SceneUniform {
    mat4 viewProjection;
    vec4 lightDirectionAmbient;
    vec4 lightColorIntensity;
    vec4 cameraPosition;
} scene;

layout(set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 tint;
    vec4 materialSpecularShininess;
} objectData;

void main()
{
    vec4 worldPos = objectData.model * vec4(inPosition, 1.0);

    gl_Position = scene.viewProjection * worldPos;

    mat3 normalMatrix =
        transpose(inverse(mat3(objectData.model)));

    worldNormal = normalize(normalMatrix * inNormal);
    worldPosition = worldPos.xyz;
    vertexColor = inColor * objectData.tint.rgb;
}