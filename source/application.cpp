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
struct PushConstants {
    float mvp[16];
    float tint[4];
};

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
Mat4 perspectiveMatrix(float aspect) {
    const float f = 1.0f / std::tan(PI / 6.0f);
    Mat4 m{};
    m[0] = f / aspect;
    m[5] = -f; // Vulkan viewport Y direction
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

bool createPipeline() {
    auto& ctx = graphics::internal::context;
    vertex_shader = createShader("shaders/torus.vert.spv");
    fragment_shader = createShader("shaders/torus.frag.spv");
    if (!vertex_shader || !fragment_shader) return false;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
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
    if (!createPipeline()) { std::cerr << "Failed to create torus graphics pipeline. Check shaders and glslc.\n"; return false; }
    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    if (vertex_shader) vkDestroyShaderModule(ctx.device, vertex_shader, nullptr);
    if (fragment_shader) vkDestroyShaderModule(ctx.device, fragment_shader, nullptr);
    if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_allocation);
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
    ImGui::Checkbox("Animate", &animation);
    ImGui::SliderFloat("Animation speed", &animation_speed, 0.0f, 3.0f);
    ImGui::SliderFloat("Orbit radius", &orbit_radius, 0.0f, 2.0f);
    if (ImGui::Button("Reset transforms")) {
        position[0] = position[1] = position[2] = 0.0f;
        rotation[0] = rotation[1] = rotation[2] = 0.0f;
        scale[0] = scale[1] = scale[2] = 1.0f;
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
    Mat4 view = translation(0.0f, 0.0f, -5.0f);
    Mat4 proj = perspective ? perspectiveMatrix(aspect) : orthoMatrix(aspect);
    Mat4 mvp = multiply(proj, multiply(view, model));
    PushConstants pc{};
    std::copy(mvp.begin(), mvp.end(), pc.mvp);
    pc.tint[0] = tint[0]; pc.tint[1] = tint[1]; pc.tint[2] = tint[2]; pc.tint[3] = 1.0f;
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, vertex_count, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application
