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
};
struct ObjectData {
    float mvp[16];
    float tint[4];
};

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;
VkBuffer uniform_buffer = VK_NULL_HANDLE;
VmaAllocation uniform_allocation = VK_NULL_HANDLE;
VkDeviceSize uniform_buffer_stride = 0;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_sets[2]{};
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkShaderModule vertex_shader = VK_NULL_HANDLE;
VkShaderModule fragment_shader = VK_NULL_HANDLE;
VkBuffer vertex_buffer_static = VK_NULL_HANDLE;
VmaAllocation vertex_allocation_static = VK_NULL_HANDLE;
float static_color_center[3] = {0.2f, 0.1f, 0.1f};
float static_color_edge[3]   = {0.8f, 0.2f, 0.2f};

float position[3] = {0.0f, 0.0f, 0.0f};
float rotation[3] = {0.0f, 0.0f, 0.0f};
float scale[3] = {1.0f, 1.0f, 1.0f};
float tint[3] = {0.25f, 0.75f, 1.0f};
bool perspective = true;
bool animation = true;
float animation_speed = 3.0f;
double previous_time = 0.0;
float animation_time = 0.0f;
float orbit_radius = 1.0f;


float tilt_degrees = 30.0f;
float orbit_angle = 0.0f;
bool orbit_enabled = true;

float static_position[3] = {2.5f, 0.0f, 0.0f};
float static_rotation[3] = {0.0f, 0.0f, 0.0f};
float static_scale[3]   = {0.6f, 0.6f, 0.6f};
float static_tint[3]    = {1.0f, 0.6f, 0.25f};
float static_brightness = 3.0f;


float proc_color_center[3] = {0.35f, 0.35f, 0.35f};
float proc_color_edge[3]   = {1.00f, 1.00f, 1.00f};
int proc_color_mode = 0;
bool vertex_colors_dirty = false;

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
Mat4 perspectiveMatrix(float aspect) {
    const float f = 1.0f / std::tan(PI / 6.0f);
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

std::array<float, 3> computeVertexColor(const Vertex& v) {
    const float maxR = MAJOR_RADIUS + MINOR_RADIUS;
    float nx = (v.position[0] / maxR + 1.0f) * 0.5f;
    float ny = (v.position[1] / MINOR_RADIUS + 1.0f) * 0.5f;
    float nz = (v.position[2] / maxR + 1.0f) * 0.5f;

    float t = 0.0f;
    if (proc_color_mode == 0) {
        t = (nx + ny + nz) / 3.0f;
    } else {
        float dx = v.position[0];
        float dy = v.position[1];
        float dz = v.position[2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        t = (dist / maxR); // 0..1
    }
    return {
        proc_color_center[0] + (proc_color_edge[0] - proc_color_center[0]) * t,
        proc_color_center[1] + (proc_color_edge[1] - proc_color_center[1]) * t,
        proc_color_center[2] + (proc_color_edge[2] - proc_color_center[2]) * t
    };
}

bool createTorus() {
    std::vector<Vertex> vertices_dynamic;
    std::vector<Vertex> vertices_static;
    vertices_dynamic.reserve(MAJOR_SEGMENTS * MINOR_SEGMENTS * 6);
    vertices_static.reserve(MAJOR_SEGMENTS * MINOR_SEGMENTS * 6);

    // Лямбда для генерации вершин с переданными параметрами цвета
    auto makeVertex = [](float u, float v, float* c_center, float* c_edge, int mode) {
        const float cu = std::cos(u), su = std::sin(u);
        const float cv = std::cos(v), sv = std::sin(v);
        Vertex out{};
        out.position[0] = (MAJOR_RADIUS + MINOR_RADIUS * cv) * cu;
        out.position[1] = MINOR_RADIUS * sv;
        out.position[2] = (MAJOR_RADIUS + MINOR_RADIUS * cv) * su;
        
        // Вычисление процедурного цвета
        const float maxR = MAJOR_RADIUS + MINOR_RADIUS;
        float nx = (out.position[0] / maxR + 1.0f) * 0.5f;
        float ny = (out.position[1] / MINOR_RADIUS + 1.0f) * 0.5f;
        float nz = (out.position[2] / maxR + 1.0f) * 0.5f;

        float t = 0.0f;
        if (mode == 0) {
            t = (nx + ny + nz) / 3.0f;
        } else {
            float dist = std::sqrt(out.position[0]*out.position[0] + 
                                   out.position[1]*out.position[1] + 
                                   out.position[2]*out.position[2]);
            t = (dist / maxR); 
        }
        out.color[0] = c_center[0] + (c_edge[0] - c_center[0]) * t;
        out.color[1] = c_center[1] + (c_edge[1] - c_center[1]) * t;
        out.color[2] = c_center[2] + (c_edge[2] - c_center[2]) * t;
        
        return out;
    };

    for (uint32_t i = 0; i < MAJOR_SEGMENTS; ++i) {
        const float u0 = 2.0f * PI * i / MAJOR_SEGMENTS;
        const float u1 = 2.0f * PI * (i + 1) / MAJOR_SEGMENTS;
        for (uint32_t j = 0; j < MINOR_SEGMENTS; ++j) {
            const float v0 = 2.0f * PI * j / MINOR_SEGMENTS;
            const float v1 = 2.0f * PI * (j + 1) / MINOR_SEGMENTS;
            
            // Генерируем для первого (динамического) тора
            Vertex a1 = makeVertex(u0, v0, proc_color_center, proc_color_edge, proc_color_mode);
            Vertex b1 = makeVertex(u1, v0, proc_color_center, proc_color_edge, proc_color_mode);
            Vertex c1 = makeVertex(u1, v1, proc_color_center, proc_color_edge, proc_color_mode);
            Vertex d1 = makeVertex(u0, v1, proc_color_center, proc_color_edge, proc_color_mode);
            vertices_dynamic.insert(vertices_dynamic.end(), {a1, b1, c1, a1, c1, d1});

            // Генерируем для второго (статического) тора
            Vertex a2 = makeVertex(u0, v0, static_color_center, static_color_edge, proc_color_mode);
            Vertex b2 = makeVertex(u1, v0, static_color_center, static_color_edge, proc_color_mode);
            Vertex c2 = makeVertex(u1, v1, static_color_center, static_color_edge, proc_color_mode);
            Vertex d2 = makeVertex(u0, v1, static_color_center, static_color_edge, proc_color_mode);
            vertices_static.insert(vertices_static.end(), {a2, b2, c2, a2, c2, d2});
        }
    }
    
    vertex_count = static_cast<uint32_t>(vertices_dynamic.size());
    auto& ctx = graphics::internal::context;

    // Создаем буферы, если их еще нет
    if (!vertex_buffer) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = vertices_dynamic.size() * sizeof(Vertex);
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        
        vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &vertex_buffer, &vertex_allocation, nullptr);
        vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &vertex_buffer_static, &vertex_allocation_static, nullptr);
    }

    // Заполняем первый буфер
    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, vertex_allocation, &mapped);
    std::memcpy(mapped, vertices_dynamic.data(), vertices_dynamic.size() * sizeof(Vertex));
    vmaUnmapMemory(ctx.allocator, vertex_allocation);

    // Заполняем второй буфер
    vmaMapMemory(ctx.allocator, vertex_allocation_static, &mapped);
    std::memcpy(mapped, vertices_static.data(), vertices_static.size() * sizeof(Vertex));
    vmaUnmapMemory(ctx.allocator, vertex_allocation_static);

    return true;
}

bool createObjectDescriptors() {
    auto& ctx = graphics::internal::context;

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(ctx.physical_device, &deviceProperties);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        1, deviceProperties.limits.minUniformBufferOffsetAlignment);
    uniform_buffer_stride =
        ((sizeof(ObjectData) + alignment - 1) / alignment) * alignment;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = uniform_buffer_stride * 2;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                        &uniform_buffer, &uniform_allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create object uniform buffer\n";
        return false;
    }

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(ctx.device, &layoutInfo, nullptr,
                                    &descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create object descriptor set layout\n";
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr,
                               &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create object descriptor pool\n";
        return false;
    }

    const VkDescriptorSetLayout layouts[] = {
        descriptor_set_layout, descriptor_set_layout
    };
    VkDescriptorSetAllocateInfo setInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptor_pool;
    setInfo.descriptorSetCount = 2;
    setInfo.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(ctx.device, &setInfo, descriptor_sets) != VK_SUCCESS) {
        std::cerr << "Failed to allocate object descriptor sets\n";
        return false;
    }

    VkDescriptorBufferInfo bufferInfos[2]{};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        bufferInfos[i].buffer = uniform_buffer;
        bufferInfos[i].offset = uniform_buffer_stride * i;
        bufferInfos[i].range = sizeof(ObjectData);

        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = descriptor_sets[i];
        writes[i].dstBinding = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    return true;
}

bool createPipeline() {
    auto& ctx = graphics::internal::context;
    vertex_shader = createShader("shaders/torus.vert.spv");
    fragment_shader = createShader("shaders/torus.frag.spv");
    if (!vertex_shader || !fragment_shader) return false;

    VkDescriptorSetLayout setLayouts[] = {descriptor_set_layout};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = setLayouts;
    if (vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipeline_layout) != VK_SUCCESS) return false;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vertex_shader; stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fragment_shader; stages[1].pName = "main";

    VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attributes[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, position))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))}
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1; vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2; vertexInput.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1; viewport.scissorCount = 1;
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2; dynamic.pDynamicStates = dynamicStates;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL; raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE; depth.depthWriteEnable = VK_TRUE; depth.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &blendAttachment;
    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2; pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput; pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewport; pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample; pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend; pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipeline_layout; pipelineInfo.renderPass = ctx.render_pass; pipelineInfo.subpass = 0;
    return vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) == VK_SUCCESS;
}
} // namespace

bool initialize() {
    if (!createTorus()) { std::cerr << "Failed to create torus vertex buffer\n"; return false; }
    if (!createObjectDescriptors()) {
        shutdown();
        return false;
    }
    if (!createPipeline()) {
        std::cerr << "Failed to create torus graphics pipeline. Check shaders and glslc.\n";
        shutdown();
        return false;
    }
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    if (descriptor_pool) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
    if (descriptor_set_layout) vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);
    if (vertex_shader) vkDestroyShaderModule(ctx.device, vertex_shader, nullptr);
    if (fragment_shader) vkDestroyShaderModule(ctx.device, fragment_shader, nullptr);
    if (uniform_buffer) vmaDestroyBuffer(ctx.allocator, uniform_buffer, uniform_allocation);
    if (vertex_buffer_static) vmaDestroyBuffer(ctx.allocator, vertex_buffer_static, vertex_allocation_static);
    if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_allocation);
}

void update(double time) {
    if (previous_time == 0.0) previous_time = time;
    float delta = static_cast<float>(std::clamp(time - previous_time, 0.0, 0.1));
    previous_time = time;
    if (animation) {
        animation_time += delta * animation_speed;
        rotation[1] += delta * animation_speed * 50.0f;
        if (orbit_enabled) {
            orbit_angle += delta * animation_speed * 0.8f;
            constexpr float TWO_PI = 2.0f * PI;
            orbit_angle = std::fmod(orbit_angle, TWO_PI);
        if (orbit_angle < 0.0f) orbit_angle += TWO_PI;
    }
        rotation[0] = tilt_degrees;
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
    if (ImGui::CollapsingHeader("Procedural Vertex Colors")) {
        if (ImGui::Combo("Mode", &proc_color_mode, "Axis gradient\0Distance gradient\0")) {
            vertex_colors_dirty = true;
        }
        if (ImGui::ColorEdit3("Color Center", proc_color_center)) {
            vertex_colors_dirty = true;
        }
        if (ImGui::ColorEdit3("Color Edge", proc_color_edge)) {
            vertex_colors_dirty = true;
        }
        ImGui::TextWrapped("Note: Final color = Tint * VertexColor");
    }
    ImGui::Separator();
    ImGui::Checkbox("Animate", &animation);
    ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 3.0f);
    ImGui::SliderFloat("Orbit_radius", &orbit_radius, -10.0f, 10.0f);
    ImGui::SliderFloat("Tilt (deg)", &tilt_degrees, -180.0f, 180.0f);
    ImGui::Checkbox("Orbit motion", &orbit_enabled);
    ImGui::SliderFloat("Orbit phase", &orbit_angle, 0.0f, 6.2831853f);
    if (ImGui::Button("Reset transforms")) {
        position[0] = position[1] = position[2] = 0.0f;
        rotation[0] = rotation[1] = rotation[2] = 0.0f;
        scale[0] = scale[1] = scale[2] = 1.0f;
        tilt_degrees = 30.0f;
        orbit_angle  = 0.0f;
    }
    ImGui::Separator();
    ImGui::Text("Second (static) torus");
    ImGui::SliderFloat3("Pos##static",   static_position, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rot##static",   static_rotation, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale##static", static_scale,     0.1f, 2.5f);
    ImGui::ColorEdit3 ("Color##static",  static_tint);
    ImGui::SliderFloat("Brightness##static", &static_brightness, 1.0f, 10.0f);
    if (vertex_colors_dirty) {
        createTorus();
        vertex_colors_dirty = false;
    }
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    if (!fd.command_buffer || !fd.framebuffer || !pipeline) return;
    auto& ctx = graphics::internal::context;

    const float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                         std::max(1.0f, static_cast<float>(ctx.swapchain_extent.height));
    const float ox = orbit_enabled ? orbit_radius * std::cos(orbit_angle) : 0.0f;
    const float oz = orbit_enabled ? orbit_radius * std::sin(orbit_angle) : 0.0f;

    Mat4 orbitT = translation(ox, 0.0f, oz);
    Mat4 userT  = translation(position[0], position[1], position[2]);
    Mat4 rotM   = multiply(rotateZ(rotation[2] * PI / 180.0f),
                multiply(rotateY(rotation[1] * PI / 180.0f),
                multiply(rotateX(rotation[0] * PI / 180.0f),
                        scaling(scale[0], scale[1], scale[2]))));
    Mat4 model = multiply(orbitT, multiply(userT, rotM));
    Mat4 view = translation(0.0f, 0.0f, -5.0f);
    Mat4 proj = perspective ? perspectiveMatrix(aspect) : orthoMatrix(aspect);
    Mat4 mvp = multiply(proj, multiply(view, model));

    Mat4 staticModel = multiply(translation(static_position[0], static_position[1], static_position[2]),
                        multiply(rotateZ(static_rotation[2] * PI / 180.0f),
                        multiply(rotateY(static_rotation[1] * PI / 180.0f),
                        multiply(rotateX(static_rotation[0] * PI / 180.0f),
                                scaling(static_scale[0], static_scale[1], static_scale[2])))));
    Mat4 staticMvp = multiply(proj, multiply(view, staticModel));

    ObjectData objectData[2]{};
    std::copy(mvp.begin(), mvp.end(), objectData[0].mvp);
    objectData[0].tint[0] = tint[0];
    objectData[0].tint[1] = tint[1];
    objectData[0].tint[2] = tint[2];
    objectData[0].tint[3] = 1.0f;
    std::copy(staticMvp.begin(), staticMvp.end(), objectData[1].mvp);
    objectData[1].tint[0] = static_tint[0] * static_brightness;
    objectData[1].tint[1] = static_tint[1] * static_brightness;
    objectData[1].tint[2] = static_tint[2] * static_brightness;
    objectData[1].tint[3] = 1.0f;

    bool objectDataReady = false;
    void* mapped = nullptr;
    if (vmaMapMemory(ctx.allocator, uniform_allocation, &mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map object uniform buffer\n";
    } else {
        auto* mappedBytes = static_cast<uint8_t*>(mapped);
        std::memcpy(mappedBytes, &objectData[0], sizeof(ObjectData));
        std::memcpy(mappedBytes + uniform_buffer_stride, &objectData[1], sizeof(ObjectData));
        if (vmaFlushAllocation(ctx.allocator, uniform_allocation, 0,
                               uniform_buffer_stride * 2) != VK_SUCCESS) {
            std::cerr << "Failed to flush object uniform buffer\n";
        } else {
            objectDataReady = true;
        }
        vmaUnmapMemory(ctx.allocator, uniform_allocation);
    }

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
    vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width = static_cast<float>(ctx.swapchain_extent.width);
    vp.height = static_cast<float>(ctx.swapchain_extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, ctx.swapchain_extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (objectDataReady) {
        VkDeviceSize offset = 0;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                0, 1, &descriptor_sets[0], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
        vkCmdDraw(cmd, vertex_count, 1, 0, 0);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                0, 1, &descriptor_sets[1], 0, nullptr);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer_static, &offset);
        vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
