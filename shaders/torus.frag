#version 450
layout(location = 0) in vec3 vertexColor;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 worldPosition;

layout(location = 0) out vec4 outColor;

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

struct PointLight {
    vec4 position;
    vec4 colorIntensity;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLightsBuffer {
    PointLight lights[];
} pointLights;

void main()
{
    vec3 N = normalize(worldNormal);
    vec3 V = normalize(scene.cameraPosition.xyz - worldPosition);
    vec3 albedo = vertexColor;

    // Ambient lighting.
    vec3 finalColor =
        albedo * scene.lightDirectionAmbient.w;

    // Directional light: Blinn–Phong.
    vec3 directionalL =
        normalize(-scene.lightDirectionAmbient.xyz);

    float directionalDiffuse = max(dot(N, directionalL), 0.0);
    vec3 directionalH = normalize(directionalL + V);

    float directionalSpecular = 0.0;
    if (directionalDiffuse > 0.0) {
        directionalSpecular = pow(
            max(dot(N, directionalH), 0.0),
            objectData.materialSpecularShininess.w
        );
    }

    vec3 directionalRadiance =
        scene.lightColorIntensity.rgb
        * scene.lightColorIntensity.w;

    finalColor +=
        albedo * directionalRadiance * directionalDiffuse;

    finalColor +=
        objectData.materialSpecularShininess.rgb
        * directionalRadiance
        * directionalSpecular;

    // Point lights: inverse-square attenuation.
    for (uint i = 0; i < pointLights.lights.length(); ++i) {
        PointLight light = pointLights.lights[i];

        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.01);
        vec3 L = toLight * inversesqrt(distanceSquared);

        float attenuation = 1.0 / distanceSquared;
        float diffuseFactor = max(dot(N, L), 0.0);

        vec3 H = normalize(L + V);
        float specularFactor = 0.0;

        if (diffuseFactor > 0.0) {
            specularFactor = pow(
                max(dot(N, H), 0.0),
                objectData.materialSpecularShininess.w
            );
        }

        vec3 radiance =
            light.colorIntensity.rgb
            * light.colorIntensity.w
            * attenuation;

        finalColor += albedo * radiance * diffuseFactor;

        finalColor +=
            objectData.materialSpecularShininess.rgb
            * radiance
            * specularFactor;
    }

    outColor = vec4(finalColor, objectData.tint.a);
}