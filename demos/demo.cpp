/*
    This demo demonstrates how to perform closest point queries using FCPW.
    Refer to the fcpw.h header file for the full API, including how to perform
    distance and ray intersection queries on a collection of query points as
    well as a single query point.

    The demo can be run from the command line using the following commands:
    > mkdir build
    > cd build
    > cmake -DFCPW_BUILD_DEMO=ON [-DFCPW_ENABLE_GPU_SUPPORT=ON] ..
    > make -j4
    > ./demos/demo [--useGpu]
*/

#include <chrono>

#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/meshio.h"

#include <fcpw/fcpw.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#ifdef FCPW_USE_GPU
    #include <fcpw/fcpw_gpu.h>
#endif

std::filesystem::path currentDirectory = std::filesystem::current_path().parent_path();
std::string g_mesh_name = (currentDirectory / "demos" / "dragon.obj").string();
std::vector<size_t> g_source_idx = {19732, 25383};

#ifdef FCPW_POLYSCOPE
    #include "polyscope/polyscope.h"
    #include "polyscope/surface_mesh.h"
    #include "polyscope/point_cloud.h"
    #include "polyscope/curve_network.h"
#endif

#include "args/args.hxx"

using namespace fcpw;

// helper class for loading polygons from obj files
struct Index {
    Index() {}
    Index(int v) : position(v) {}

    bool operator<(const Index& i) const {
        if (position < i.position) return true;
        if (position > i.position) return false;

        return false;
    }

    int position;
};

// helper function for loading polygons from obj files
inline Index parseFaceIndex(const std::string& token) {
    std::stringstream in(token);
    std::string indexString;
    int indices[3] = {1, 1, 1};

    int i = 0;
    while (std::getline(in, indexString, '/')) {
        if (indexString != "\\") {
            std::stringstream ss(indexString);
            ss >> indices[i++];
        }
    }

    // decrement since indices in OBJ files are 1-based
    return Index(indices[0] - 1);
}

void loadObj(const std::string& objFilePath,
             std::vector<Vector<3>>& positions,
             std::vector<Vector3i>& indices)
{
    // initialize
    std::ifstream in(objFilePath);
    if (in.is_open() == false) {
        std::cerr << "Unable to open file: " << objFilePath << std::endl;
        exit(EXIT_FAILURE);
    }

    // parse
    std::string line;
    positions.clear();
    indices.clear();

    while (getline(in, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;

            positions.emplace_back(Vector3(x, y, z));
        
        } else if (token == "f") {
            int j = 0;
            Vector3i face;

            while (ss >> token) {
                Index index = parseFaceIndex(token);

                if (index.position < 0) {
                    getline(in, line);
                    size_t i = line.find_first_not_of("\t\n\v\f\r ");
                    index = parseFaceIndex(line.substr(i));
                }

                face[j++] = index.position;
            }

            indices.emplace_back(face);
        }
    }

    // close
    in.close();
}

void loadFcpwScene(const std::vector<Vector<3>>& positions,
                   const std::vector<Vector3i>& indices,
                   bool buildVectorizedBvh, Scene<3>& scene)
{
    // load positions and indices
    scene.setObjectCount(1);
    scene.setObjectVertices(positions, 0);
    scene.setObjectTriangles(indices, 0);

    // build scene on CPU
    AggregateType aggregateType = AggregateType::Bvh_SurfaceArea;
    bool printStats = false;
    bool reduceMemoryFootprint = false;
    scene.build(aggregateType, buildVectorizedBvh, printStats, reduceMemoryFootprint);
}

template <typename T>
void performClosestPointQueries(const std::vector<Vector<3>>& queryPoints,
                                std::vector<Vector<3>>& closestPoints,
                                T& scene)
{
    // do nothing
}

template <>
void performClosestPointQueries(const std::vector<Vector<3>>& queryPoints,
                                std::vector<Vector<3>>& closestPoints,
                                Scene<3>& scene)
{
    // initialize bounding spheres
    std::vector<BoundingSphere<3>> boundingSpheres;
    for (const Vector<3>& q: queryPoints) {
        boundingSpheres.emplace_back(BoundingSphere<3>(q, maxFloat));
    }

    // perform cpqs
    std::vector<Interaction<3>> interactions;
    scene.findClosestPoints(boundingSpheres, interactions);

    // extract closest points
    closestPoints.clear();
    for (const Interaction<3>& i: interactions) {
        closestPoints.emplace_back(i.p);
    }
}

#ifdef FCPW_USE_GPU

template <>
void performClosestPointQueries(const std::vector<Vector<3>>& queryPoints,
                                std::vector<Vector<3>>& closestPoints,
                                GPUScene<3>& gpuScene)
{
    // initialize bounding spheres
    std::vector<GPUBoundingSphere> boundingSpheres;
    for (const Vector<3>& q: queryPoints) {
        float3 queryPoint = float3{q[0], q[1], q[2]};
        boundingSpheres.emplace_back(GPUBoundingSphere(queryPoint, maxFloat));
    }

    // perform cpqs on GPU
    std::vector<GPUInteraction> interactions;
    gpuScene.findClosestPoints(boundingSpheres, interactions);

    // extract closest points
    closestPoints.clear();
    for (const GPUInteraction& i: interactions) {
        closestPoints.emplace_back(Vector<3>(i.p.x, i.p.y, i.p.z));
    }
}


void performGeoPath(const std::vector<Vector<3>>& queryPoints,
                                std::vector<Vector<3>>& closestPoints,
                                GPUScene<3>& gpuScene)
{
    std::cout << queryPoints.size() << std::endl;
    std::cout << closestPoints.size() << std::endl;

    // initialize bounding spheres
    std::vector<GPUBoundingSphere> inputBoundingSpheres;
    std::vector<GPUBoundingSphere> outputBoundingSpheres;
    for (const Vector<3>& q: queryPoints) {
        float3 queryPoint = float3{q[0], q[1], q[2]};
        inputBoundingSpheres.emplace_back(GPUBoundingSphere(queryPoint, maxFloat));
    }
    outputBoundingSpheres = inputBoundingSpheres;

    gpuScene.computeGeoPath(inputBoundingSpheres, outputBoundingSpheres);

    // extract closest points
    closestPoints.clear();
    for (const GPUBoundingSphere& b: outputBoundingSpheres) {
        closestPoints.emplace_back(Vector<3>(b.c.x, b.c.y, b.c.z));
    }
}

#endif

// distance between two points u and v, i.e., Euclidean distance ||u - v||_2
float Distance(const Vector<3> &u, const Vector<3> &v)
{
    assert(u.size() == v.size());

    float distance = 0.0;
    for(size_t d = 0; d < u.size(); ++d) 
    {
        distance += pow(u[d] - v[d], 2);
    }
    distance = sqrt(distance);

    return distance;
}

size_t GetGeodesicPath(Vector<3> &p_start, Vector<3> &p_end, size_t max_iters, std::vector<Vector<3>> &initial_path, std::vector<Vector<3>> &path, Scene<3>& scene)
{   
    //////////////////////////////////////////////////////////
    // now compute geodesic path via harmonic map
    //////////////////////////////////////////////////////////

    // float dist_max = 0.001;

    // NOTE: we do not need adaptivity because the initial path is always in the mesh surface when constructed by Dijkstra's algorithm
    // std::cout << "Before adaptivity = " << initial_path.size() << std::endl;
    // SpatialAdaptivity(surface_tube, initial_path, dist_max);
    // std::cout << "After adaptivity = " << initial_path.size() << std::endl;

// #ifdef FCPW_POLYSCOPE
//     if(m_is_closed)
//     {
//         polyscope::registerCurveNetworkLoop("inital refined cp path", initial_path)->setRadius(0.004)->setColor(init_blue);
//     }
//     else
//     {
//         polyscope::registerCurveNetworkLine("inital refined cp path", initial_path)->setRadius(0.004)->setColor(init_blue);
//     }
// #endif

    path = initial_path;

    float path_diff = 1.0;
    size_t iter = 0;
    float mu = 0.2; // dt / dx^2, with dt = 0.5 dx^2
    while(path_diff > 1e-8 && iter < max_iters)
    {
        ++iter;
        // std::cout << "Iteration " << iter << std::endl;
        // cout << "Path size = " << initial_path.size() << endl;

        // discretize a 1D line [0,1]
        // independent heat flow for each coordinate along the 1D line M (explicit Euler)
        for(size_t i = 1; i < path.size() - 1; ++i)
        {
            for(size_t d = 0; d < 3; ++d)
            { 
                path[i][d] = initial_path[i][d] + mu * (initial_path[i-1][d] - 2.0 * initial_path[i][d] + initial_path[i+1][d]);
            }
        }

        bool m_is_closed = false;
        if(m_is_closed) // apply periodic BCs
        {
            for(size_t d = 0; d < 3; ++d)
            { 
                path[0][d] = initial_path[0][d] + mu * (initial_path[path.size() - 1][d] - 2.0 * initial_path[0][d] + initial_path[1][d]);
                path[path.size() - 1][d] = initial_path[path.size() - 1][d] + mu * (initial_path[path.size() - 2][d] - 2.0 * initial_path[path.size() - 1][d] + initial_path[0][d]);
            }
        }

        // project points back onto target surface N
        std::vector<Vector<3>> cp_path;
        performClosestPointQueries(path, cp_path, scene);

        path_diff = 0.0;
        for(size_t i = 0; i < cp_path.size(); ++i)
        {
            path_diff += Distance(initial_path[i], cp_path[i]);
        }  
        path_diff /= cp_path.size();

        // SpatialAdaptivity(surface_tube, path, dist_max);

        initial_path = cp_path;
    }

    return iter;
}

template <typename T>
void guiCallback(std::vector<Vector<3>>& queryPoints,
                 std::vector<Vector<3>>& closestPoints,
                 T& scene)
{
    // animate query points
    for (Vector<3>& q: queryPoints) {
        q[0] += 0.001 * std::sin(10.0 * q[1]);
        q[1] += 0.001 * std::cos(10.0 * q[0]);
    }

    // perform closest point queries
    performClosestPointQueries(queryPoints, closestPoints, scene);

    // plot results
#ifdef FCPW_POLYSCOPE
    polyscope::registerPointCloud("query points", queryPoints);
    polyscope::registerPointCloud("closest points", closestPoints);
#endif

    std::vector<Vector2i> edgeIndices;
    std::vector<Vector<3>> edgePositions = queryPoints;
    edgePositions.insert(edgePositions.end(), closestPoints.begin(), closestPoints.end());
    for (int i = 0; i < (int)queryPoints.size(); i++) {
        edgeIndices.emplace_back(Vector2i(i, i + queryPoints.size()));
    }

#ifdef FCPW_POLYSCOPE
    auto network = polyscope::registerCurveNetwork("edges", edgePositions, edgeIndices);
    network->setRadius(0.005, false);
#endif
}

template <typename T>
void visualize(const std::vector<Vector<3>>& positions,
               const std::vector<Vector3i>& indices,
               std::vector<Vector<3>>& queryPoints,
               std::vector<Vector<3>>& closestPoints,
               T& scene)
{
#ifdef FCPW_POLYSCOPE
    // set a few options
    polyscope::options::programName = "FCPW Demo";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    // register mesh and callback
    polyscope::registerSurfaceMesh("mesh", positions, indices);
    polyscope::state::userCallback = std::bind(&guiCallback<T>, std::ref(queryPoints),
                                               std::ref(closestPoints), std::ref(scene));

    // give control to polyscope gui
    polyscope::show();
#endif
}

void run(bool useGpu)
{
    // load obj file
    std::vector<Vector<3>> positions;
    std::vector<Vector3i> indices;
    loadObj(g_mesh_name, positions, indices);

    // generate random query points for closest point queries
    Vector<3> boxMin = Vector<3>::Constant(std::numeric_limits<float>::infinity());
    Vector<3> boxMax = -Vector<3>::Constant(std::numeric_limits<float>::infinity());
    for (const Vector<3>& p: positions) {
        boxMin = boxMin.cwiseMin(p);
        boxMax = boxMax.cwiseMax(p);
    }

    int numQueryPoints = 100;
    std::vector<Vector<3>> queryPoints;
    Vector<3> boxExtent = boxMax - boxMin;
    for (int i = 0; i < numQueryPoints; i++) {
        queryPoints.emplace_back(boxMin + boxExtent.cwiseProduct(uniformRealRandomVector<3>()));
    }

    // Load a mesh
    std::unique_ptr<geometrycentral::surface::ManifoldSurfaceMesh> mesh;
    std::unique_ptr<geometrycentral::surface::VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = geometrycentral::surface::readManifoldSurfaceMesh(g_mesh_name);

    // Create a path network as a Dijkstra path between endpoints
    std::unique_ptr<geometrycentral::surface::FlipEdgeNetwork> edgeNetwork;
    geometrycentral::surface::Vertex vStart = mesh->vertex(g_source_idx[0]);
    geometrycentral::surface::Vertex vEnd = mesh->vertex(g_source_idx[1]);
    edgeNetwork = geometrycentral::surface::FlipEdgeNetwork::constructFromDijkstraPath(*mesh, *geometry, vStart, vEnd);

    edgeNetwork->posGeom = geometry.get();
    std::vector<std::vector<geometrycentral::Vector3>> init_polyline = edgeNetwork->getPathPolyline3D();
    std::vector<Vector<3>> initial_path(init_polyline[0].size());
    for(size_t i = 0; i < initial_path.size(); ++i)
    {
        for(size_t d = 0; d < 3; ++d)
        {
            initial_path[i][d] = init_polyline[0][i][d];
        }
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Make the path a geodesic
    edgeNetwork->iterativeShorten();

    // Extract the result as a polyline along the surface
    edgeNetwork->posGeom = geometry.get();
    std::vector<std::vector<geometrycentral::Vector3>> polyline = edgeNetwork->getPathPolyline3D();

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    std::cout << "Flip Geo path total elapsed time: " << elapsed_seconds.count() << "s\n";

#ifdef FCPW_POLYSCOPE
    // initialize polyscope
    polyscope::init();

    polyscope::registerSurfaceMesh("mesh", geometry->vertexPositions, mesh->getFaceVertexList());
    polyscope::registerCurveNetworkLine("Initial Path", initial_path);
    polyscope::registerCurveNetworkLine("Flip Geodesic Path", polyline[0]);
#endif

    if (useGpu) {
#ifdef FCPW_USE_GPU
        // load fcpw scene on CPU
        Scene<3> scene;
        loadFcpwScene(positions, indices, false, scene); // NOTE: must build non-vectorized CPU BVH

        // transfer scene to GPU
        bool printStats = false;
        GPUScene<3> gpuScene(currentDirectory.string(), printStats);
        gpuScene.transferToGPU(scene);

        std::vector<Vector<3>> path;
        size_t max_iters = 10000;

        start = std::chrono::system_clock::now();

        performGeoPath(initial_path, path, gpuScene);
        std::cout << "Path size = " << path.size() << std::endl;
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Geo path total elapsed time: " << elapsed_seconds.count() << "s\n";

    #ifdef FCPW_POLYSCOPE
        polyscope::registerCurveNetworkLine("Flip Geo Path", polyline[0]);
        polyscope::registerSurfaceMesh("Surface", positions, indices)->setSmoothShade(true);
        polyscope::registerCurveNetworkLine("Path", path);
    #endif

        float length = 0.0;
        for(size_t i = 0; i < path.size() - 1; ++i)
        {
            float sum_diff_sqrd = 0.0;
            for(size_t d = 0; d < 3; ++d)
            {
                sum_diff_sqrd += pow(path[i+1][d] - path[i][d], 2);
            }
            length += sqrt(sum_diff_sqrd);
        }

        std::cout << "Harmonic total length = " << length << std::endl;

#ifdef FCPW_POLYSCOPE
        polyscope::registerCurveNetworkLine("CP Path", path);

        polyscope::show();
#endif

        // // visualize results
        // std::vector<Vector<3>> closestPoints;
        // visualize(positions, indices, queryPoints, closestPoints, gpuScene);
#else
        std::cerr << "GPU support not enabled" << std::endl;
        exit(EXIT_FAILURE);
#endif

    } else {
        // load fcpw scene
        Scene<3> scene;
        loadFcpwScene(positions, indices, true, scene);

        // cp approach
        Vector<3> p_start;
        Vector<3> p_end;
        for(size_t d = 0; d < 3; ++d)
        {
            p_start[d] = geometry->vertexPositions[g_source_idx[0]][d];
            p_end[d] = geometry->vertexPositions[g_source_idx[1]][d];
        }

        std::vector<Vector<3>> path;
        size_t max_iters = 10000;

        start = std::chrono::system_clock::now();

        GetGeodesicPath(p_start, p_end, max_iters, initial_path, path, scene);
        
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        std::cout << "Geo path total elapsed time: " << elapsed_seconds.count() << "s\n";

    #ifdef FCPW_POLYSCOPE
        polyscope::registerCurveNetworkLine("Flip Geo Path", polyline[0]);
        polyscope::registerSurfaceMesh("Surface", positions, indices)->setSmoothShade(true);
        polyscope::registerCurveNetworkLine("Path", path);
    #endif

        float length = 0.0;
        for(size_t i = 0; i < path.size() - 1; ++i)
        {
            float sum_diff_sqrd = 0.0;
            for(size_t d = 0; d < 3; ++d)
            {
                sum_diff_sqrd += pow(path[i+1][d] - path[i][d], 2);
            }
            length += sqrt(sum_diff_sqrd);
        }

        std::cout << "Harmonic total length = " << length << std::endl;

#ifdef FCPW_POLYSCOPE
        polyscope::registerCurveNetworkLine("CP Path", path);

        polyscope::show();
#endif

        // // visualize results
        // std::vector<Vector<3>> closestPoints;
        // visualize(positions, indices, queryPoints, closestPoints, scene);
    }
}

int main(int argc, const char *argv[]) {
    // configure the argument parser
    args::ArgumentParser parser("fcpw demo");
    args::Group group(parser, "", args::Group::Validators::DontCare);
    args::Flag useGpu(group, "bool", "use GPU", {"useGpu"});

    // parse args
    try {
        parser.ParseCLI(argc, argv);

    } catch (const args::Help&) {
        std::cout << parser;
        return 0;

    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    run(args::get(useGpu));

    return 0;
}