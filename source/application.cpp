#include "application.hpp"

#include <imgui.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

// cmake -S . -B build-vs -G "Visual Studio 17 2022" -A x64
// cmake --build build-vs --config Debug --parallel 4
namespace application {
namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t MAJOR_SEGMENTS = 64;
constexpr uint32_t MINOR_SEGMENTS = 24;
constexpr float MAJOR_RADIUS = 1.0f;
constexpr float MINOR_RADIUS = 0.38f;

struct Vertex {
    float position[3];
    float color[3];
    float normal[3];
};
struct PushConstants {
    float mvp[16];
    float tint[4];
    float lightDirectionAmbient[4];
    float lightColorIntensity[4];
    float materialSpecularShininess[4];
};

struct alignas(16) SceneUniform {
    float viewProjection[16];
    float lightDirectionAmbient[4];
    float lightColorIntensity[4];
    float cameraPosition[4];
};

struct alignas(16) ModelUniform {
    float model[16];
    float tint[4];
    float materialSpecularShininess[4];
};

struct alignas(16) PointLight
{
    float position[4];
    float colorIntensity[4];
};

VkBuffer point_lights_buffer = VK_NULL_HANDLE;
VmaAllocation point_lights_allocation = VK_NULL_HANDLE;

uint32_t point_lights_capacity = 0;

std::vector<PointLight> point_lights = {
    {
        { 2.0f, 1.5f, 2.0f, 1.0f },
        { 1.0f, 0.25f, 0.12f, 8.0f }
    },
    {
        { -2.0f, 0.5f, 1.0f, 1.0f },
        { 0.15f, 0.4f, 1.0f, 6.0f }
    }
};

VkBuffer scene_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation scene_uniform_allocation = VK_NULL_HANDLE;

VkBuffer model_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation model_uniform_allocation = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;

VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkShaderModule vertex_shader = VK_NULL_HANDLE;
VkShaderModule fragment_shader = VK_NULL_HANDLE;

float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {0.0f, 0.0f, 0.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};
float tint[3] = {0.25f, 0.75f, 1.0f};
bool perspective = true;
bool animation = true;
float lightDirection[3] = {-0.4f, -0.8f, -0.3f};
float lightColor[3] = {1.0f, 0.95f, 0.85f};
float lightIntensity = 1.0f;
float ambientIntensity = 0.18f;
float specularColor[3] = {0.8f, 0.85f, 1.0f};
float shininess = 32.0f;
float cameraPosition[3] = {0.0f, 0.0f, 5.0f};
float cameraRotation[2] = {0.0f, 0.0f};
float cameraFov = 60.0f;

float animation_speed = 3.0f;
float orbit_radius = 1.0f;
double previous_time = 0.0;
float animation_time = 0.0f;

using Mat4 = std::array<float, 16>;
Mat4 identity() {
    Mat4 m{};
    m[0] = m[5] = m[10] = m[15] = 1.0f;
    return m;
}
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                r[c * 4 + row] += a[k * 4 + row] * b[c * 4 + k];
    return r;
}
Mat4 translation(float x, float y, float z) {
    Mat4 m = identity(); m[12] = x; m[13] = y; m[14] = z; return m;
}
Mat4 scaling(float x, float y, float z) {
    Mat4 m{}; m[0] = x; m[5] = y; m[10] = z; m[15] = 1.0f; return m;
}
Mat4 rotateX(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m[5] = c; m[9] = -s; m[6] = s; m[10] = c; return m;
}
Mat4 rotateY(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m[0] = c; m[8] = s; m[2] = -s; m[10] = c; return m;
}
Mat4 rotateZ(float a) {
    Mat4 m = identity(); float c = std::cos(a), s = std::sin(a);
    m[0] = c; m[4] = -s; m[1] = s; m[5] = c; return m;
}
Mat4 perspectiveMatrix(float aspect, float fovDegrees) {
    const float fovRadians = fovDegrees * PI / 180.0f;
    const float f = 1.0f / std::tan(fovRadians * 0.5f);

    Mat4 m{};
    m[0] = f / aspect;
    m[5] = -f;
    m[10] = 100.0f / (0.1f - 100.0f);
    m[11] = -1.0f;
    m[14] = (100.0f * 0.1f) / (0.1f - 100.0f);
    return m;
}
Mat4 orthoMatrix(float aspect) {
    Mat4 m = identity();
    const float halfHeight = 2.5f;
    const float halfWidth = halfHeight * aspect;
    m[0] = 1.0f / halfWidth;
    m[5] = -1.0f / halfHeight;
    m[10] = 1.0f / (0.1f - 100.0f);
    m[14] = 0.1f / (0.1f - 100.0f);
    return m;
}

std::vector<uint32_t> readSpirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open shader: " << path << '\n'; return {}; }
    const auto size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}
VkShaderModule createShader(const char* path) {
    auto code = readSpirv(path);
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * sizeof(uint32_t);
    ci.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &module) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return module;
}

bool createTorus() {
    std::vector<Vertex> vertices;
    vertices.reserve(MAJOR_SEGMENTS * MINOR_SEGMENTS * 6);
    auto makeVertex = [](float u, float v) {
        const float cu = std::cos(u), su = std::sin(u);
        const float cv = std::cos(v), sv = std::sin(v);
        Vertex out{};
        out.position[0] = (MAJOR_RADIUS + MINOR_RADIUS * cv) * cu;
        out.position[1] = MINOR_RADIUS * sv;
        out.position[2] = (MAJOR_RADIUS + MINOR_RADIUS * cv) * su;
        out.normal[0] = cv * cu;
        out.normal[1] = sv;
        out.normal[2] = cv * su;
        // Procedural vertex colors derived from the local position.
        out.color[0] = 0.35f + 0.65f * (out.position[0] / (MAJOR_RADIUS + MINOR_RADIUS) + 1.0f) * 0.5f;
        out.color[1] = 0.35f + 0.65f * (out.position[1] / MINOR_RADIUS + 1.0f) * 0.5f;
        out.color[2] = 0.35f + 0.65f * (out.position[2] / (MAJOR_RADIUS + MINOR_RADIUS) + 1.0f) * 0.5f;
        return out;
    };
    for (uint32_t i = 0; i < MAJOR_SEGMENTS; ++i) {
        const float u0 = 2.0f * PI * i / MAJOR_SEGMENTS;
        const float u1 = 2.0f * PI * (i + 1) / MAJOR_SEGMENTS;
        for (uint32_t j = 0; j < MINOR_SEGMENTS; ++j) {
            const float v0 = 2.0f * PI * j / MINOR_SEGMENTS;
            const float v1 = 2.0f * PI * (j + 1) / MINOR_SEGMENTS;
            Vertex a = makeVertex(u0, v0), b = makeVertex(u1, v0);
            Vertex c = makeVertex(u1, v1), d = makeVertex(u0, v1);
            vertices.insert(vertices.end(), {a, b, c, a, c, d});
        }
    }
    vertex_count = static_cast<uint32_t>(vertices.size());
    auto& ctx = graphics::internal::context;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = vertices.size() * sizeof(Vertex);
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &vertex_buffer, &vertex_allocation, nullptr) != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    if (vmaMapMemory(ctx.allocator, vertex_allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(Vertex));
    vmaUnmapMemory(ctx.allocator, vertex_allocation);
    return true;
}

bool createUniformBuffer(
    VkDeviceSize size,
    VkBuffer& buffer,
    VmaAllocation& allocation)
{
    auto& ctx = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
    };
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    return vmaCreateBuffer(
        ctx.allocator,
        &bufferInfo,
        &allocInfo,
        &buffer,
        &allocation,
        nullptr
    ) == VK_SUCCESS;
}

bool createPointLightsBuffer(uint32_t capacity)
{
    auto& ctx = graphics::internal::context;

    capacity = std::max(capacity, 1u);

    VkBufferCreateInfo bufferInfo{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
    };
    bufferInfo.size =
        static_cast<VkDeviceSize>(capacity) * sizeof(PointLight);
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VkBuffer newBuffer = VK_NULL_HANDLE;
    VmaAllocation newAllocation = VK_NULL_HANDLE;

    if (vmaCreateBuffer(
            ctx.allocator,
            &bufferInfo,
            &allocInfo,
            &newBuffer,
            &newAllocation,
            nullptr) != VK_SUCCESS)
    {
        return false;
    }

    // Переносим данные в новый буфер до замены старого.
    if (!point_lights.empty())
    {
        void* mapped = nullptr;
        if (vmaMapMemory(
                ctx.allocator, newAllocation, &mapped) != VK_SUCCESS)
        {
            vmaDestroyBuffer(ctx.allocator, newBuffer, newAllocation);
            return false;
        }

        const size_t bytes =
            point_lights.size() * sizeof(PointLight);
        std::memcpy(mapped, point_lights.data(), bytes);
        vmaFlushAllocation(ctx.allocator, newAllocation, 0, bytes);
        vmaUnmapMemory(ctx.allocator, newAllocation);
    }

    if (point_lights_buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(
            ctx.allocator,
            point_lights_buffer,
            point_lights_allocation);
    }

    point_lights_buffer = newBuffer;
    if (scene_descriptor_set != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo info{};
        info.buffer = point_lights_buffer;
        info.offset = 0;
        info.range =
            static_cast<VkDeviceSize>(
                point_lights_capacity) * sizeof(PointLight);

        VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        };
        write.dstSet = scene_descriptor_set;
        write.dstBinding = 1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    }
    point_lights_allocation = newAllocation;
    point_lights_capacity = capacity;

    return true;
}

bool uploadPointLights()
{
    if (point_lights.size() > point_lights_capacity)
    {
        const uint32_t newCapacity = std::max(
            point_lights_capacity * 2,
            static_cast<uint32_t>(point_lights.size()));

        if (!createPointLightsBuffer(newCapacity))
            return false;
    }

    // Буфер всегда содержит место минимум для одного элемента.
    PointLight emptyLight{};
    const void* source = point_lights.empty()
        ? static_cast<const void*>(&emptyLight)
        : static_cast<const void*>(point_lights.data());

    const size_t bytes = std::max(
        sizeof(PointLight),
        point_lights.size() * sizeof(PointLight));

    void* mapped = nullptr;
    auto& ctx = graphics::internal::context;

    if (vmaMapMemory(
            ctx.allocator, point_lights_allocation, &mapped)
        != VK_SUCCESS)
    {
        return false;
    }

    std::memcpy(mapped, source, bytes);
    vmaFlushAllocation(
        ctx.allocator, point_lights_allocation, 0, bytes);
    vmaUnmapMemory(ctx.allocator, point_lights_allocation);

    return true;
}

bool writeUniformBuffer(
    VkBuffer buffer,
    VmaAllocation allocation,
    const void* data,
    VkDeviceSize size)
{
    auto& ctx = graphics::internal::context;

    void* mapped = nullptr;
    if (vmaMapMemory(ctx.allocator, allocation, &mapped)
        != VK_SUCCESS)
    {
        return false;
    }

    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(ctx.allocator, allocation);
    return true;
}

bool createDescriptorResources()
{
    auto& ctx = graphics::internal::context;

    VkDescriptorSetLayoutBinding sceneBindings[2]{};

    // set = 0, binding = 0: параметры сцены.
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (!createPointLightsBuffer(
        static_cast<uint32_t>(point_lights.size())))
    {
        return false;
    }
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;

    // set = 0, binding = 1: массив точечных источников.
    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    sceneLayoutInfo.bindingCount = 2;
    sceneLayoutInfo.pBindings = sceneBindings;

    if (vkCreateDescriptorSetLayout(
            ctx.device,
            &sceneLayoutInfo,
            nullptr,
            &scene_set_layout) != VK_SUCCESS)
    {
        return false;
    }

    VkDescriptorSetLayoutBinding modelBinding{};
    modelBinding.binding = 0;
    modelBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    modelBinding.descriptorCount = 1;
    modelBinding.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo modelLayoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    };
    modelLayoutInfo.bindingCount = 1;
    modelLayoutInfo.pBindings = &modelBinding;

    if (vkCreateDescriptorSetLayout(
            ctx.device, &modelLayoutInfo, nullptr,
            &model_set_layout) != VK_SUCCESS)
    {
        return false;
    }

    if (!createUniformBuffer(
            sizeof(SceneUniform),
            scene_uniform_buffer,
            scene_uniform_allocation))
    {
        return false;
    }

    if (!createUniformBuffer(
            sizeof(ModelUniform),
            model_uniform_buffer,
            model_uniform_allocation))
    {
        return false;
    }

    VkDescriptorPoolSize poolSizes[2]{};

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 2;

    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    };
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(
            ctx.device, &poolInfo, nullptr,
            &descriptor_pool) != VK_SUCCESS)
    {
        return false;
    }

    VkDescriptorSetLayout layouts[] = {
        scene_set_layout,
        model_set_layout
    };

    VkDescriptorSet sets[2]{};

    VkDescriptorSetAllocateInfo allocInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    };
    allocInfo.descriptorPool = descriptor_pool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(
            ctx.device, &allocInfo, sets) != VK_SUCCESS)
    {
        return false;
    }

    scene_descriptor_set = sets[0];
    model_descriptor_set = sets[1];

    VkDescriptorBufferInfo sceneBufferInfo{};
    sceneBufferInfo.buffer = scene_uniform_buffer;
    sceneBufferInfo.offset = 0;
    sceneBufferInfo.range = sizeof(SceneUniform);

    VkDescriptorBufferInfo modelBufferInfo{};
    modelBufferInfo.buffer = model_uniform_buffer;
    modelBufferInfo.offset = 0;
    modelBufferInfo.range = sizeof(ModelUniform);

    VkDescriptorBufferInfo pointLightsInfo{};
    pointLightsInfo.buffer = point_lights_buffer;
    pointLightsInfo.offset = 0;
    pointLightsInfo.range =
        static_cast<VkDeviceSize>(
            point_lights_capacity) * sizeof(PointLight);

    VkWriteDescriptorSet writes[3]{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = scene_descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &sceneBufferInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = model_descriptor_set;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &modelBufferInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = scene_descriptor_set;
    writes[2].dstBinding = 1;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &pointLightsInfo;

    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    return true;
}




bool createPipeline()
{
    auto& ctx = graphics::internal::context;
    vertex_shader = createShader("shaders/torus.vert.spv");
    fragment_shader = createShader("shaders/torus.frag.spv");

    if (!vertex_shader || !fragment_shader)
        return false;

    // Pipeline layout: два descriptor set, без push constants.
    VkDescriptorSetLayout setLayouts[] = {
        scene_set_layout, // set = 0: параметры сцены
        model_set_layout  // set = 1: параметры модели
    };

    VkPipelineLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };

    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(
            ctx.device,
            &layoutInfo,
            nullptr,
            &pipeline_layout) != VK_SUCCESS)
    {
        return false;
    }

    // Shader stages.
    VkPipelineShaderStageCreateInfo stages[2]{};

    stages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_shader;
    stages[0].pName = "main";

    stages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragment_shader;
    stages[1].pName = "main";

    // Vertex input.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[3]{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset =
        static_cast<uint32_t>(offsetof(Vertex, position));

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset =
        static_cast<uint32_t>(offsetof(Vertex, color));

    attributes[2].location = 2;
    attributes[2].binding = 0;
    attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[2].offset =
        static_cast<uint32_t>(offsetof(Vertex, normal));

    VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
    };
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3;
    vertexInput.pVertexAttributeDescriptions = attributes;

    // Input assembly.
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
    };
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assembly.primitiveRestartEnable = VK_FALSE;

    // Dynamic viewport and scissor.
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
    };
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
    };
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    // Rasterization.
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
    };
    raster.depthClampEnable = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = VK_FALSE;
    raster.lineWidth = 1.0f;

    // Multisampling.
    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
    };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;

    // Depth testing.
    VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
    };
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;
    depth.depthBoundsTestEnable = VK_FALSE;
    depth.stencilTestEnable = VK_FALSE;

    // Color blending.
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
    };
    blend.logicOpEnable = VK_FALSE;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    // Graphics pipeline.
    VkGraphicsPipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO
    };

    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipeline_layout;
    pipelineInfo.renderPass = ctx.render_pass;
    pipelineInfo.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(
        ctx.device,
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline
    );

    if (result != VK_SUCCESS)
    {
        pipeline = VK_NULL_HANDLE;
        return false;
    }

    return true;
}
} // namespace

bool initialize()
{
    if (!createTorus()) {std::cerr << "Failed to create torus vertex buffer\n";return false;}
    if (!createDescriptorResources()) {std::cerr << "Failed to create descriptor resources\n"; return false;}
    if (!createPipeline()) {std::cerr
            << "Failed to create torus graphics pipeline. "
            << "Check shaders and glslc.\n";
        return false;}
    return true;
}

void shutdown()
{
    auto& ctx = graphics::internal::context;

    if (ctx.device != VK_NULL_HANDLE) {
        vkQueueWaitIdle(ctx.graphics_queue);

        if (pipeline) {
            vkDestroyPipeline(ctx.device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }

        if (pipeline_layout) {
            vkDestroyPipelineLayout(
                ctx.device, pipeline_layout, nullptr);
            pipeline_layout = VK_NULL_HANDLE;
        }

        if (vertex_shader) {
            vkDestroyShaderModule(
                ctx.device, vertex_shader, nullptr);
            vertex_shader = VK_NULL_HANDLE;
        }

        if (fragment_shader) {
            vkDestroyShaderModule(
                ctx.device, fragment_shader, nullptr);
            fragment_shader = VK_NULL_HANDLE;
        }

        if (descriptor_pool) {
            vkDestroyDescriptorPool(
                ctx.device, descriptor_pool, nullptr);
            descriptor_pool = VK_NULL_HANDLE;
        }

        if (scene_set_layout) {
            vkDestroyDescriptorSetLayout(
                ctx.device, scene_set_layout, nullptr);
            scene_set_layout = VK_NULL_HANDLE;
        }

        if (model_set_layout) {
            vkDestroyDescriptorSetLayout(
                ctx.device, model_set_layout, nullptr);
            model_set_layout = VK_NULL_HANDLE;
        }
    }

    if (ctx.allocator != VK_NULL_HANDLE) {
        if (scene_uniform_buffer) {
            vmaDestroyBuffer(
                ctx.allocator,
                scene_uniform_buffer,
                scene_uniform_allocation);

            scene_uniform_buffer = VK_NULL_HANDLE;
            scene_uniform_allocation = VK_NULL_HANDLE;
        }

        if (model_uniform_buffer) {
            vmaDestroyBuffer(
                ctx.allocator,
                model_uniform_buffer,
                model_uniform_allocation);

            model_uniform_buffer = VK_NULL_HANDLE;
            model_uniform_allocation = VK_NULL_HANDLE;
        }

        if (vertex_buffer) {
            vmaDestroyBuffer(
                ctx.allocator,
                vertex_buffer,
                vertex_allocation);

            vertex_buffer = VK_NULL_HANDLE;
            vertex_allocation = VK_NULL_HANDLE;
        }
    }
}

void update(double time) {
    if (previous_time == 0.0) previous_time = time;
    float delta = static_cast<float>(std::clamp(time - previous_time, 0.0, 0.1));
    previous_time = time;
    if (animation) {
        animation_time += delta * animation_speed;

        rotation[0] = 90.0f;
        rotation[1] += delta * animation_speed * 50.0f;
        rotation[2] = 0.0f;
    }
    ImGui::Begin("Torus controls");
    ImGui::Text("Variant 7: Torus");
    ImGui::Separator();
    ImGui::Text("Projection");
    if (ImGui::RadioButton("Perspective", perspective))
        perspective = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", !perspective))
        perspective = false;
    ImGui::Separator();
    ImGui::Text("Position");
    ImGui::SliderFloat3("XYZ##position", position, -3.0f, 3.0f);
    ImGui::Text("Rotation (degrees)");
    ImGui::SliderFloat3("XYZ##rotation", rotation, -180.0f, 180.0f);
    ImGui::Text("Scale");
    ImGui::SliderFloat3("XYZ##scale", scale, 0.1f, 2.5f);
    ImGui::Separator();
    ImGui::ColorEdit3("Object color", tint);
    ImGui::Separator();
    ImGui::Text("Directional light");
    ImGui::SliderFloat3(
        "Light direction",
        lightDirection,
        -1.0f,
        1.0f
    );
    ImGui::ColorEdit3("Light color", lightColor);
    ImGui::SliderFloat(
        "Light intensity",
        &lightIntensity,
        0.0f,
        5.0f
    );
    ImGui::SliderFloat(
        "Ambient intensity",
        &ambientIntensity,
        0.0f,
        1.0f
    );
    ImGui::Separator();
    ImGui::Text("Material");
    ImGui::ColorEdit3("Specular color", specularColor);
    ImGui::SliderFloat(
        "Shininess",
        &shininess,
        1.0f,
        128.0f
    );
    ImGui::Separator();
    ImGui::Checkbox("Animate", &animation);
    ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 3.0f);
    ImGui::SliderFloat("Orbit radius", &orbit_radius, 0.0f, 2.0f);
    if (ImGui::Button("Reset transforms")) {
        position[0] = position[1] = position[2] = 0.0f;
        rotation[0] = rotation[1] = rotation[2] = 0.0f;
        scale[0] = scale[1] = scale[2] = 1.0f;
    }
    ImGui::Separator();
    ImGui::Text("Camera");

    ImGui::SliderFloat3("Camera position", cameraPosition, -10.0f, 10.0f);
    ImGui::SliderFloat("Camera pitch", &cameraRotation[0], -89.0f, 89.0f);
    ImGui::SliderFloat("Camera yaw", &cameraRotation[1], -180.0f, 180.0f);

    if (perspective) {
        ImGui::SliderFloat("Field of view", &cameraFov, 20.0f, 100.0f);
    }
    ImGui::Separator();
    ImGui::Text("Point lights");

    if (ImGui::Button("Add point light")) {
        PointLight light{};
        light.position[0] = 0.0f;
        light.position[1] = 1.5f;
        light.position[2] = 2.0f;
        light.position[3] = 1.0f;

        light.colorIntensity[0] = 1.0f;
        light.colorIntensity[1] = 1.0f;
        light.colorIntensity[2] = 1.0f;
        light.colorIntensity[3] = 5.0f;

        point_lights.push_back(light);
    }

    for (size_t i = 0; i < point_lights.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));

        ImGui::Text("Point light %d", static_cast<int>(i + 1));
        ImGui::SliderFloat3(
            "Position",
            point_lights[i].position,
            -5.0f, 5.0f);

        ImGui::ColorEdit3(
            "Color",
            point_lights[i].colorIntensity);

        ImGui::SliderFloat(
            "Intensity",
            &point_lights[i].colorIntensity[3],
            0.0f, 50.0f);

        if (ImGui::Button("Remove")) {
            point_lights.erase(point_lights.begin() + i);
            ImGui::PopID();
            break;
        }

        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    if (!fd.command_buffer || !fd.framebuffer || !pipeline) return;
    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = fd.command_buffer;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) return;

    VkClearValue clears[2]{};
    clears[0].color = {{0.035f, 0.045f, 0.07f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = ctx.render_pass; pass.framebuffer = fd.framebuffer;
    pass.renderArea.extent = ctx.swapchain_extent;
    pass.clearValueCount = 2; pass.pClearValues = clears;
    if (!uploadPointLights()) {
        std::cerr << "Failed to upload point lights\n";
        return;
    }
    vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width = static_cast<float>(ctx.swapchain_extent.width);
    vp.height = static_cast<float>(ctx.swapchain_extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);

    const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                         std::max(1.0f, static_cast<float>(ctx.swapchain_extent.height));
    Mat4 model = multiply(translation(position[0], position[1], position[2]),
                multiply(rotateZ(rotation[2] * PI / 180.0f),
                multiply(rotateY(rotation[1] * PI / 180.0f),
                multiply(rotateX(rotation[0] * PI / 180.0f), scaling(scale[0], scale[1], scale[2])))));
    const float pitch = cameraRotation[0] * PI / 180.0f;
    const float yaw = cameraRotation[1] * PI / 180.0f;

    Mat4 cameraTranslation = translation(
        -cameraPosition[0],
        -cameraPosition[1],
        -cameraPosition[2]
    );

    // Обратное вращение камеры.
    Mat4 cameraInverseRotation = multiply(rotateX(-pitch), rotateY(-yaw));

    Mat4 view = multiply(cameraInverseRotation, cameraTranslation);

    Mat4 proj = perspective ? perspectiveMatrix(aspect, cameraFov) : orthoMatrix(aspect);
    Mat4 mvp = multiply(proj, multiply(view, model));
    const Mat4 viewProjection = multiply(proj, view);

    SceneUniform sceneData{};

    std::copy(
        viewProjection.begin(),
        viewProjection.end(),
        sceneData.viewProjection
    );

    sceneData.lightDirectionAmbient[0] = lightDirection[0];
    sceneData.lightDirectionAmbient[1] = lightDirection[1];
    sceneData.lightDirectionAmbient[2] = lightDirection[2];
    sceneData.lightDirectionAmbient[3] = ambientIntensity;

    sceneData.lightColorIntensity[0] = lightColor[0];
    sceneData.lightColorIntensity[1] = lightColor[1];
    sceneData.lightColorIntensity[2] = lightColor[2];
    sceneData.lightColorIntensity[3] = lightIntensity;

    sceneData.cameraPosition[0] = cameraPosition[0];
    sceneData.cameraPosition[1] = cameraPosition[1];
    sceneData.cameraPosition[2] = cameraPosition[2];
    sceneData.cameraPosition[3] = 1.0f;

    ModelUniform modelData{};

    std::copy(
        model.begin(),
        model.end(),
        modelData.model
    );

    modelData.tint[0] = tint[0];
    modelData.tint[1] = tint[1];
    modelData.tint[2] = tint[2];
    modelData.tint[3] = 1.0f;

    modelData.materialSpecularShininess[0] = specularColor[0];
    modelData.materialSpecularShininess[1] = specularColor[1];
    modelData.materialSpecularShininess[2] = specularColor[2];
    modelData.materialSpecularShininess[3] = shininess;

    writeUniformBuffer(
        scene_uniform_buffer,
        scene_uniform_allocation,
        &sceneData,
        sizeof(sceneData)
    );

    writeUniformBuffer(
        model_uniform_buffer,
        model_uniform_allocation,
        &modelData,
        sizeof(modelData)
    );

    VkDescriptorSet descriptorSets[] = {
        scene_descriptor_set,
        model_descriptor_set
    };

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout,
        0,
        2,
        descriptorSets,
        0,
        nullptr
    );

    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
