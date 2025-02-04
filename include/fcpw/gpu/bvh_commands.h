#pragma once

#include <fcpw/gpu/bvh_interop_structures.h>

namespace fcpw {

void loadModuleLibrary(GPUContext& gpuContext,
                       const std::string& moduleLibrary,
                       Shader& shader)
{
    Slang::Result loadModuleLibraryResult = shader.loadModuleLibrary(
        gpuContext.device, moduleLibrary.c_str());

    if (loadModuleLibraryResult != SLANG_OK) {
        std::cout << "failed to load " << moduleLibrary << " module library" << std::endl;
        exit(EXIT_FAILURE);

    } else {
        std::cout << "loaded " << moduleLibrary << " module library" << std::endl;
    }
}

void loadShader(GPUContext& gpuContext,
                const std::string& shaderModule,
                const std::string& entryPointName,
                Shader& shader)
{
    Slang::Result loadComputeProgramResult = shader.loadComputeProgram(
        gpuContext.device, shaderModule.c_str(), entryPointName.c_str());

    if (loadComputeProgramResult != SLANG_OK) {
        std::cout << "failed to load " << entryPointName << " compute program" << std::endl;
        exit(EXIT_FAILURE);

    } else {
        std::cout << "loaded " << entryPointName << " compute program" << std::endl;
    }
}

template <typename T, typename S>
void runTraversal(GPUContext& gpuContext,
                  const Shader& shader,
                  const T& gpuBvhBuffers,
                  const S& gpuQueryBuffers,
                  std::vector<GPUInteraction>& interactions,
                  int nThreadGroups = 4096,
                  bool printLogs = false)
{
    // setup command buffer
    auto commandBuffer = gpuContext.transientHeap->createCommandBuffer();
    auto encoder = commandBuffer->encodeComputeCommands();

    // create shader objects
    auto rootShaderObject = encoder->bindPipeline(shader.pipelineState);
    ComPtr<IShaderObject> bvhShaderObject = gpuBvhBuffers.createShaderObject(
                                                gpuContext.device, shader, printLogs);
    ShaderCursor rootCursor(rootShaderObject);
    rootCursor.getPath("gBvh").setObject(bvhShaderObject);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    int entryPointFieldCount = gpuQueryBuffers.setResources(entryPointCursor);
    if (printLogs) {
        std::cout << "runTraversal" << std::endl;
        for (int i = 0; i < entryPointFieldCount; i++) {
            std::cout << "\tcursor[" << i << "]: " << entryPointCursor.getTypeLayout()->getFieldByIndex(i)->getName() << std::endl;
        }
    }

    // perform query
    ComPtr<IQueryPool> queryPool;
    IQueryPool::Desc queryDesc = {};
    queryDesc.type = QueryType::Timestamp;
    queryDesc.count = 2;
    Slang::Result createQueryPoolResult = gpuContext.device->createQueryPool(queryDesc, queryPool.writeRef());
    if (createQueryPoolResult != SLANG_OK) {
        std::cout << "failed to create query pool" << std::endl;
        exit(EXIT_FAILURE);
    }

    encoder->writeTimestamp(queryPool, 0);
    encoder->dispatchCompute(nThreadGroups, 1, 1);
    encoder->writeTimestamp(queryPool, 1);

    // execute command buffer
    encoder->endEncoding();
    commandBuffer->close();
    gpuContext.queue->executeCommandBuffer(commandBuffer);
    gpuContext.queue->waitOnHost();

    // read query timestamps
    const DeviceInfo& deviceInfo = gpuContext.device->getDeviceInfo();
    double timestampFrequency = (double)deviceInfo.timestampFrequency;
    uint64_t timestampData[2] = { 0, 0 };
    Slang::Result getQueryPoolResult = queryPool->getResult(0, 2, timestampData);
    if (getQueryPoolResult != SLANG_OK) {
        std::cout << "failed to get query pool result" << std::endl;
        exit(EXIT_FAILURE);
    }

    // read back results from GPU
    gpuQueryBuffers.read(gpuContext.device, interactions);

    // synchronize and reset transient heap
    gpuContext.transientHeap->finish();
    gpuContext.transientHeap->synchronizeAndReset();

    if (printLogs) {
        double timeSpan = (timestampData[1] - timestampData[0])*1000/timestampFrequency;
        std::cout << interactions.size() << " queries"
                  << " took " << timeSpan << " ms"
                  << std::endl;
    }
}

template <typename T, typename S>
void runTraversal(GPUContext& gpuContext,
                  const Shader& shader,
                  const T& gpuBvhBuffers,
                  const S& gpuQueryBuffers,
                  std::vector<GPUBoundingSphere>& boundingSpheres,
                  bool& continueIteration,
                  int nThreadGroups = 4096,
                  bool printLogs = false)
{
    // setup command buffer
    auto commandBuffer = gpuContext.transientHeap->createCommandBuffer();
    auto encoder = commandBuffer->encodeComputeCommands();

    // create shader objects
    auto rootShaderObject = encoder->bindPipeline(shader.pipelineState);
    ComPtr<IShaderObject> bvhShaderObject = gpuBvhBuffers.createShaderObject(
                                                gpuContext.device, shader, printLogs);
    ShaderCursor rootCursor(rootShaderObject);
    rootCursor.getPath("gBvh").setObject(bvhShaderObject);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    int entryPointFieldCount = gpuQueryBuffers.setResources(entryPointCursor);
    if (printLogs) {
        std::cout << "runTraversal" << std::endl;
        for (int i = 0; i < entryPointFieldCount; i++) {
            std::cout << "\tcursor[" << i << "]: " << entryPointCursor.getTypeLayout()->getFieldByIndex(i)->getName() << std::endl;
        }
    }

    // perform query
    ComPtr<IQueryPool> queryPool;
    IQueryPool::Desc queryDesc = {};
    queryDesc.type = QueryType::Timestamp;
    queryDesc.count = 2;
    Slang::Result createQueryPoolResult = gpuContext.device->createQueryPool(queryDesc, queryPool.writeRef());
    if (createQueryPoolResult != SLANG_OK) {
        std::cout << "failed to create query pool" << std::endl;
        exit(EXIT_FAILURE);
    }

    encoder->writeTimestamp(queryPool, 0);
    encoder->dispatchCompute(nThreadGroups, 1, 1);
    encoder->writeTimestamp(queryPool, 1);

    // execute command buffer
    encoder->endEncoding();
    commandBuffer->close();
    gpuContext.queue->executeCommandBuffer(commandBuffer);
    gpuContext.queue->waitOnHost();

    // read query timestamps
    const DeviceInfo& deviceInfo = gpuContext.device->getDeviceInfo();
    double timestampFrequency = (double)deviceInfo.timestampFrequency;
    uint64_t timestampData[2] = { 0, 0 };
    Slang::Result getQueryPoolResult = queryPool->getResult(0, 2, timestampData);
    if (getQueryPoolResult != SLANG_OK) {
        std::cout << "failed to get query pool result" << std::endl;
        exit(EXIT_FAILURE);
    }

    // read back results from GPU
    // if(!continueIteration)
    // {
        // std::cout << "here in final iteration" << std::endl;
        gpuQueryBuffers.read(gpuContext.device, boundingSpheres);
        continueIteration = false;
    
        // synchronize and reset transient heap
        gpuContext.transientHeap->finish();
        gpuContext.transientHeap->synchronizeAndReset();
    // }

    if (printLogs) {
        double timeSpan = (timestampData[1] - timestampData[0])*1000/timestampFrequency;
        std::cout << boundingSpheres.size() << " queries"
                  << " took " << timeSpan << " ms"
                  << std::endl;
    }
}


template <typename T>
void runUpdate(GPUContext& gpuContext,
               const Shader& shader,
               const T& gpuBvhBuffers,
               bool printLogs = false)
{
    // setup command buffer
    auto commandBuffer = gpuContext.transientHeap->createCommandBuffer();
    auto encoder = commandBuffer->encodeComputeCommands();

    // create shader objects
    auto rootShaderObject = encoder->bindPipeline(shader.pipelineState);
    ComPtr<IShaderObject> bvhShaderObject = gpuBvhBuffers.createShaderObject(
                                                gpuContext.device, shader, printLogs);
    ShaderCursor rootCursor(rootShaderObject);
    rootCursor.getPath("gBvh").setObject(bvhShaderObject);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    entryPointCursor.getPath("nodeIndices").setResource(gpuBvhBuffers.nodeIndices.view);

    for (int depth = gpuBvhBuffers.maxUpdateDepth; depth >= 0; --depth) {
        uint32_t firstNodeOffset = gpuBvhBuffers.updateEntryData[depth].first;
        uint32_t nodeCount = gpuBvhBuffers.updateEntryData[depth].second;
        entryPointCursor.getPath("firstNodeOffset").setData(firstNodeOffset);
        entryPointCursor.getPath("nodeCount").setData(nodeCount);

        encoder->dispatchCompute(nodeCount, 1, 1);
        encoder->bufferBarrier(gpuBvhBuffers.nodes.buffer.get(), ResourceState::UnorderedAccess, ResourceState::UnorderedAccess);
    }

    if (printLogs) {
        std::cout << "runUpdate" << std::endl;
        int entryPointFieldCount = 4;
        for (int i = 0; i < entryPointFieldCount; i++) {
            std::cout << "\tcursor[" << i << "]: " << entryPointCursor.getTypeLayout()->getFieldByIndex(i)->getName() << std::endl;
        }
    }

    // execute command buffer
    encoder->endEncoding();
    commandBuffer->close();
    gpuContext.queue->executeCommandBuffer(commandBuffer);
    gpuContext.queue->waitOnHost();

    // synchronize and reset transient heap
    gpuContext.transientHeap->finish();
    gpuContext.transientHeap->synchronizeAndReset();
}

} // namespace fcpw