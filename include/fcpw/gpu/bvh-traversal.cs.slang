import fcpw;

#define UNDEFINED_BVH_TYPE 0
#define LINE_SEGMENT_BVH 1
#define TRIANGLE_BVH 2
#define LINE_SEGMENT_SNCH 3
#define TRIANGLE_SNCH 4

#ifndef _BVH_TYPE
#define _BVH_TYPE UNDEFINED_BVH_TYPE
#endif

#if _BVH_TYPE == LINE_SEGMENT_BVH
uniform ParameterBlock<Bvh<BvhNode, LineSegment, NoSilhouette>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 0

#elif _BVH_TYPE == TRIANGLE_BVH
uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 0

#elif _BVH_TYPE == LINE_SEGMENT_SNCH
uniform ParameterBlock<Bvh<SnchNode, LineSegment, Vertex>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 1

#elif _BVH_TYPE == TRIANGLE_SNCH
uniform ParameterBlock<Bvh<SnchNode, Triangle, Edge>> gBvh;
#define _BVH_HAS_SILHOUETTE_DATA 1

#else
// Compile time error
#error _BVH_TYPE is not set to a supported type
#endif

groupshared Atomic<float> gPathDiff;

[shader("compute")]
[numthreads(256, 1, 1)]
void rayIntersection(uint3 threadId: SV_DispatchThreadID,
                     uniform StructuredBuffer<Ray> rays,
                     uniform bool checkForOcclusion,
                     uniform RWStructuredBuffer<Interaction> interactions,
                     uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Ray r = rays[index];
    Interaction i;
    bool didIntersect = gBvh.intersect(r, checkForOcclusion, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sphereIntersection(uint3 threadId: SV_DispatchThreadID,
                        uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                        uniform StructuredBuffer<float3> randNums,
                        uniform RWStructuredBuffer<Interaction> interactions,
                        uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    BoundingSphere s = boundingSpheres[index];
    float3 randNum = randNums[index];
    ConstantBranchTraversalWeight branchTraversalWeight;
    Interaction i;
    bool didIntersect = gBvh.intersect<ConstantBranchTraversalWeight>(s, randNum, branchTraversalWeight, i);
    if (didIntersect)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void closestPoint(uint3 threadId: SV_DispatchThreadID,
                  uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                  uniform RWStructuredBuffer<Interaction> interactions,
                  uniform bool recordNormals,
                  uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    BoundingSphere s = boundingSpheres[index];
    Interaction i;
    bool found = gBvh.findClosestPoint(s, i, recordNormals);
    if (found)
    {
        interactions[index] = i;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void geoPath(uint3 threadId: SV_DispatchThreadID,
             uniform RWStructuredBuffer<BoundingSphere> inputBoundingSpheres,
             uniform RWStructuredBuffer<BoundingSphere> outputBoundingSpheres,
             uniform StructuredBuffer<uint32_t> prevNbr,
             uniform StructuredBuffer<uint32_t> nextNbr,
             uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    float mu = 0.5;
    BoundingSphere s = inputBoundingSpheres[index];
    if (prevNbr[index] != 100000000 && nextNbr[index] != 200000000)
    {
        s.c.x = inputBoundingSpheres[index].c.x + mu * (inputBoundingSpheres[prevNbr[index]].c.x - 2.0f * inputBoundingSpheres[index].c.x + inputBoundingSpheres[nextNbr[index]].c.x);
        s.c.y = inputBoundingSpheres[index].c.y + mu * (inputBoundingSpheres[prevNbr[index]].c.y - 2.0f * inputBoundingSpheres[index].c.y + inputBoundingSpheres[nextNbr[index]].c.y);
        s.c.z = inputBoundingSpheres[index].c.z + mu * (inputBoundingSpheres[prevNbr[index]].c.z - 2.0f * inputBoundingSpheres[index].c.z + inputBoundingSpheres[nextNbr[index]].c.z);
    }
    else // fixed endpoints, Dirichlet condition
    {
        s = inputBoundingSpheres[index]; 
    }

    // bool m_is_closed = false;
    // if (m_is_closed) // apply periodic BCs
    // {
    //     for (size_t d = 0; d < 3; ++d)
    //     {
    //         path[0][d] = initial_path[0][d] + mu * (initial_path[path.size() - 1][d] - 2.0 * initial_path[0][d] + initial_path[1][d]);
    //         path[path.size() - 1][d] = initial_path[path.size() - 1][d] + mu * (initial_path[path.size() - 2][d] - 2.0 * initial_path[path.size() - 1][d] + initial_path[0][d]);
    //     }
    // }
    
    Interaction i;
    bool found = gBvh.findClosestPoint(s, i, false);
    if (found)
    {
        outputBoundingSpheres[index].c = i.p;
    }

    // float dist = 0.0f;
    // for (size_t dim = 0; dim < 3; ++dim)
    // {
    //     float diff = inputBoundingSpheres[index].c[dim] - outputBoundingSpheres[index].c[dim];
    //     dist += diff * diff;
    // }
    // dist = sqrt(dist);

    // gPathDiff.add(dist);
    inputBoundingSpheres[index].c = outputBoundingSpheres[index].c;
    
}

[shader("compute")]
[numthreads(256, 1, 1)]
void closestSilhouettePoint(uint3 threadId: SV_DispatchThreadID,
                            uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                            uniform StructuredBuffer<uint> flipNormalOrientation,
                            uniform float squaredMinRadius,
                            uniform float precision,
                            uniform RWStructuredBuffer<Interaction> interactions,
                            uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries)
    {
        return;
    }

    Interaction i;
#if _BVH_HAS_SILHOUETTE_DATA
    BoundingSphere s = boundingSpheres[index];
    bool flipNormal = flipNormalOrientation[index] == 1 ? true : false;
    bool found = gBvh.findClosestSilhouettePoint(s, flipNormal, squaredMinRadius, precision, i);
    if (found)
    {
        interactions[index] = i;
    }
#endif
}
