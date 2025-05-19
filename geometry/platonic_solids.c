/*
 * PLATONIC SOLIDS IN C
 * --------------------
 * This code implements all five Platonic solids:
 *   1. Tetrahedron (4 triangular faces)
 *   2. Cube/Hexahedron (6 square faces)
 *   3. Octahedron (8 triangular faces)
 *   4. Dodecahedron (12 pentagonal faces)
 *   5. Icosahedron (20 triangular faces)
 *
 * For each solid, the code provides:
 *   - Vertex coordinates
 *   - Face definitions
 *   - Edge definitions
 *   - Property calculations (volume, surface area, etc.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

/* Constants */
#define MAX_VERTICES 30
#define MAX_FACES 30
#define MAX_EDGES 60
#define MAX_VERTICES_PER_FACE 10
#define PI 3.14159265358979323846
#define PHI 1.618033988749895  // Golden ratio

/* 3D Point/Vector */
typedef struct {
    double x;
    double y;
    double z;
} Point3D;

/* Edge connecting two vertices */
typedef struct {
    int vertex1;
    int vertex2;
} Edge;

/* Face consisting of multiple vertices */
typedef struct {
    int num_vertices;
    int vertices[MAX_VERTICES_PER_FACE];
} Face;

/* Platonic Solid structure */
typedef struct {
    // Type identification
    enum {
        TETRAHEDRON,
        CUBE,
        OCTAHEDRON,
        DODECAHEDRON,
        ICOSAHEDRON
    } type;
    char name[20];
    
    // Geometry
    int num_vertices;
    Point3D vertices[MAX_VERTICES];
    
    int num_faces;
    Face faces[MAX_FACES];
    
    int num_edges;
    Edge edges[MAX_EDGES];
    
    // Properties
    int vertices_per_face;  // Number of vertices per face (3 for triangle, etc.)
    int faces_per_vertex;   // Number of faces meeting at each vertex
    double edge_length;     // Length of each edge (for unit solids)
    double dihedral_angle;  // Angle between adjacent faces (in radians)
    
    // Derived properties (filled by calculations)
    double inradius;        // Distance from center to face
    double midradius;       // Distance from center to edge midpoint
    double circumradius;    // Distance from center to vertex
    double surface_area;
    double volume;
} PlatonicSolid;

/* Function Prototypes */
// Construction functions
PlatonicSolid create_tetrahedron(double edge_length);
PlatonicSolid create_cube(double edge_length);
PlatonicSolid create_octahedron(double edge_length);
PlatonicSolid create_dodecahedron(double edge_length);
PlatonicSolid create_icosahedron(double edge_length);

// Utility functions
double distance(Point3D p1, Point3D p2);
Point3D vector_subtract(Point3D p1, Point3D p2);
double dot_product(Point3D v1, Point3D v2);
Point3D cross_product(Point3D v1, Point3D v2);
double vector_magnitude(Point3D v);
Point3D vector_normalize(Point3D v);
void generate_edges_from_faces(PlatonicSolid *solid);
void calculate_solid_properties(PlatonicSolid *solid);
double calculate_face_area(PlatonicSolid solid, int face_index);
double calculate_face_normal(PlatonicSolid solid, int face_index, Point3D *normal);
void print_solid_properties(PlatonicSolid solid);

/* Vector Operations */
double distance(Point3D p1, Point3D p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

Point3D vector_subtract(Point3D p1, Point3D p2) {
    Point3D result = {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
    return result;
}

double dot_product(Point3D v1, Point3D v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Point3D cross_product(Point3D v1, Point3D v2) {
    Point3D result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return result;
}

double vector_magnitude(Point3D v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

Point3D vector_normalize(Point3D v) {
    double mag = vector_magnitude(v);
    Point3D result = {v.x / mag, v.y / mag, v.z / mag};
    return result;
}

Point3D vector_scale(Point3D v, double scalar) {
    Point3D result = {v.x * scalar, v.y * scalar, v.z * scalar};
    return result;
}

/* Utility Functions */
// Generate edges from faces (removing duplicates)
void generate_edges_from_faces(PlatonicSolid *solid) {
    solid->num_edges = 0;
    bool edge_exists[MAX_VERTICES][MAX_VERTICES] = {false};
    
    // Loop through all faces
    for (int f = 0; f < solid->num_faces; f++) {
        Face face = solid->faces[f];
        
        // Loop through all edges in this face
        for (int v = 0; v < face.num_vertices; v++) {
            int v1 = face.vertices[v];
            int v2 = face.vertices[(v + 1) % face.num_vertices];
            
            // Ensure v1 < v2 for consistency
            if (v1 > v2) {
                int temp = v1;
                v1 = v2;
                v2 = temp;
            }
            
            // If we haven't seen this edge before, add it
            if (!edge_exists[v1][v2]) {
                solid->edges[solid->num_edges].vertex1 = v1;
                solid->edges[solid->num_edges].vertex2 = v2;
                solid->num_edges++;
                edge_exists[v1][v2] = true;
            }
        }
    }
}

// Calculate the area of a face
double calculate_face_area(PlatonicSolid solid, int face_index) {
    Face face = solid.faces[face_index];
    
    if (face.num_vertices < 3) {
        return 0.0;  // Not a valid polygon
    }
    
    // For platonic solids, all faces are regular polygons
    // We can calculate area using the formula:
    // Area = (n * s^2) / (4 * tan(π/n))
    // where n is the number of sides and s is the side length
    
    int n = face.num_vertices;
    double s = solid.edge_length;
    
    return (n * s * s) / (4 * tan(PI / n));
}

// Calculate face normal vector
double calculate_face_normal(PlatonicSolid solid, int face_index, Point3D *normal) {
    Face face = solid.faces[face_index];
    
    // Get three vertices to form vectors
    Point3D v1 = solid.vertices[face.vertices[0]];
    Point3D v2 = solid.vertices[face.vertices[1]];
    Point3D v3 = solid.vertices[face.vertices[2]];
    
    // Create two vectors on the face
    Point3D edge1 = vector_subtract(v2, v1);
    Point3D edge2 = vector_subtract(v3, v1);
    
    // Cross product gives normal vector
    *normal = cross_product(edge1, edge2);
    
    // Return magnitude and normalize
    double mag = vector_magnitude(*normal);
    normal->x /= mag;
    normal->y /= mag;
    normal->z /= mag;
    
    return mag / 2.0;  // Area of the triangle
}

// Calculate properties of a platonic solid
void calculate_solid_properties(PlatonicSolid *solid) {
    // First, check if all vertices are equidistant from origin
    double dist_sum = 0.0;
    Point3D origin = {0, 0, 0};
    
    for (int i = 0; i < solid->num_vertices; i++) {
        dist_sum += distance(solid->vertices[i], origin);
    }
    
    solid->circumradius = dist_sum / solid->num_vertices;
    
    // Calculate surface area
    solid->surface_area = 0.0;
    for (int f = 0; f < solid->num_faces; f++) {
        solid->surface_area += calculate_face_area(*solid, f);
    }
    
    // Calculate inradius (distance from center to face)
    // For regular polyhedra, all faces are equidistant from center
    Point3D face_normal;
    calculate_face_normal(*solid, 0, &face_normal);
    
    // Get a point on the face (first vertex of first face)
    Point3D face_point = solid->vertices[solid->faces[0].vertices[0]];
    
    // Project vector from origin to face point onto normal
    double inradius = fabs(dot_product(face_point, face_normal));
    solid->inradius = inradius;
    
    // Midradius (distance from center to edge midpoint)
    // Take first edge
    int v1 = solid->edges[0].vertex1;
    int v2 = solid->edges[0].vertex2;
    Point3D edge_midpoint = {
        (solid->vertices[v1].x + solid->vertices[v2].x) / 2.0,
        (solid->vertices[v1].y + solid->vertices[v2].y) / 2.0,
        (solid->vertices[v1].z + solid->vertices[v2].z) / 2.0
    };
    solid->midradius = distance(edge_midpoint, origin);
    
    // Calculate volume
    // For regular polyhedra: V = (1/3) * surface area * inradius
    solid->volume = (1.0/3.0) * solid->surface_area * solid->inradius;
}

// Print properties of a platonic solid
void print_solid_properties(PlatonicSolid solid) {
    printf("Platonic Solid: %s\n", solid.name);
    printf("Configuration: %d vertices, %d faces, %d edges\n", 
           solid.num_vertices, solid.num_faces, solid.num_edges);
    printf("Vertices per face: %d\n", solid.vertices_per_face);
    printf("Faces per vertex: %d\n", solid.faces_per_vertex);
    printf("Edge length: %.4f\n", solid.edge_length);
    printf("Dihedral angle: %.4f degrees\n", solid.dihedral_angle * 180.0 / PI);
    printf("Inradius (center to face): %.4f\n", solid.inradius);
    printf("Midradius (center to edge midpoint): %.4f\n", solid.midradius);
    printf("Circumradius (center to vertex): %.4f\n", solid.circumradius);
    printf("Surface area: %.4f\n", solid.surface_area);
    printf("Volume: %.4f\n", solid.volume);
    printf("\n");
}

/* Platonic Solid Constructions */

// Create a tetrahedron with the given edge length
PlatonicSolid create_tetrahedron(double edge_length) {
    PlatonicSolid tetra;
    strcpy(tetra.name, "Tetrahedron");
    tetra.type = TETRAHEDRON;
    tetra.edge_length = edge_length;
    
    // Basic properties
    tetra.vertices_per_face = 3;  // Triangular faces
    tetra.faces_per_vertex = 3;   // 3 faces meet at each vertex
    tetra.dihedral_angle = acos(1.0/3.0);  // ~70.53 degrees
    
    // Calculate scale factor to achieve desired edge length
    double unit_edge = sqrt(2);  // Edge length of a unit tetrahedron
    double scale = edge_length / unit_edge;
    
    // Vertex coordinates (centered at origin)
    tetra.num_vertices = 4;
    tetra.vertices[0] = (Point3D){  scale,  scale,  scale};
    tetra.vertices[1] = (Point3D){ -scale, -scale,  scale};
    tetra.vertices[2] = (Point3D){ -scale,  scale, -scale};
    tetra.vertices[3] = (Point3D){  scale, -scale, -scale};
    
    // Face definitions (CCW order)
    tetra.num_faces = 4;
    
    // Face 0: 0,1,2
    tetra.faces[0].num_vertices = 3;
    tetra.faces[0].vertices[0] = 0;
    tetra.faces[0].vertices[1] = 1;
    tetra.faces[0].vertices[2] = 2;
    
    // Face 1: 0,2,3
    tetra.faces[1].num_vertices = 3;
    tetra.faces[1].vertices[0] = 0;
    tetra.faces[1].vertices[1] = 2;
    tetra.faces[1].vertices[2] = 3;
    
    // Face 2: 0,3,1
    tetra.faces[2].num_vertices = 3;
    tetra.faces[2].vertices[0] = 0;
    tetra.faces[2].vertices[1] = 3;
    tetra.faces[2].vertices[2] = 1;
    
    // Face 3: 1,3,2
    tetra.faces[3].num_vertices = 3;
    tetra.faces[3].vertices[0] = 1;
    tetra.faces[3].vertices[1] = 3;
    tetra.faces[3].vertices[2] = 2;
    
    // Generate edges
    generate_edges_from_faces(&tetra);
    
    // Calculate other properties
    calculate_solid_properties(&tetra);
    
    return tetra;
}

// Create a cube with the given edge length
PlatonicSolid create_cube(double edge_length) {
    PlatonicSolid cube;
    strcpy(cube.name, "Cube");
    cube.type = CUBE;
    cube.edge_length = edge_length;
    
    // Basic properties
    cube.vertices_per_face = 4;  // Square faces
    cube.faces_per_vertex = 3;   // 3 faces meet at each vertex
    cube.dihedral_angle = acos(0.0);  // 90 degrees
    
    // Calculate half-edge for vertex coordinates
    double half_edge = edge_length / 2.0;
    
    // Vertex coordinates (centered at origin)
    cube.num_vertices = 8;
    cube.vertices[0] = (Point3D){ -half_edge, -half_edge, -half_edge};
    cube.vertices[1] = (Point3D){  half_edge, -half_edge, -half_edge};
    cube.vertices[2] = (Point3D){  half_edge,  half_edge, -half_edge};
    cube.vertices[3] = (Point3D){ -half_edge,  half_edge, -half_edge};
    cube.vertices[4] = (Point3D){ -half_edge, -half_edge,  half_edge};
    cube.vertices[5] = (Point3D){  half_edge, -half_edge,  half_edge};
    cube.vertices[6] = (Point3D){  half_edge,  half_edge,  half_edge};
    cube.vertices[7] = (Point3D){ -half_edge,  half_edge,  half_edge};
    
    // Face definitions (CCW order when viewed from outside)
    cube.num_faces = 6;
    
    // Face 0: Bottom (-z)
    cube.faces[0].num_vertices = 4;
    cube.faces[0].vertices[0] = 0;
    cube.faces[0].vertices[1] = 1;
    cube.faces[0].vertices[2] = 2;
    cube.faces[0].vertices[3] = 3;
    
    // Face 1: Top (+z)
    cube.faces[1].num_vertices = 4;
    cube.faces[1].vertices[0] = 4;
    cube.faces[1].vertices[1] = 7;
    cube.faces[1].vertices[2] = 6;
    cube.faces[1].vertices[3] = 5;
    
    // Face 2: Front (-y)
    cube.faces[2].num_vertices = 4;
    cube.faces[2].vertices[0] = 0;
    cube.faces[2].vertices[1] = 4;
    cube.faces[2].vertices[2] = 5;
    cube.faces[2].vertices[3] = 1;
    
    // Face 3: Back (+y)
    cube.faces[3].num_vertices = 4;
    cube.faces[3].vertices[0] = 2;
    cube.faces[3].vertices[1] = 6;
    cube.faces[3].vertices[2] = 7;
    cube.faces[3].vertices[3] = 3;
    
    // Face 4: Left (-x)
    cube.faces[4].num_vertices = 4;
    cube.faces[4].vertices[0] = 0;
    cube.faces[4].vertices[1] = 3;
    cube.faces[4].vertices[2] = 7;
    cube.faces[4].vertices[3] = 4;
    
    // Face 5: Right (+x)
    cube.faces[5].num_vertices = 4;
    cube.faces[5].vertices[0] = 1;
    cube.faces[5].vertices[1] = 5;
    cube.faces[5].vertices[2] = 6;
    cube.faces[5].vertices[3] = 2;
    
    // Generate edges
    generate_edges_from_faces(&cube);
    
    // Calculate other properties
    calculate_solid_properties(&cube);
    
    return cube;
}

// Create an octahedron with the given edge length
PlatonicSolid create_octahedron(double edge_length) {
    PlatonicSolid octa;
    strcpy(octa.name, "Octahedron");
    octa.type = OCTAHEDRON;
    octa.edge_length = edge_length;
    
    // Basic properties
    octa.vertices_per_face = 3;  // Triangular faces
    octa.faces_per_vertex = 4;   // 4 faces meet at each vertex
    octa.dihedral_angle = acos(-1.0/3.0);  // ~109.47 degrees
    
    // Calculate scale factor for desired edge length
    double unit_edge = sqrt(2);  // Edge length for unit octahedron
    double scale = edge_length / unit_edge;
    
    // Vertex coordinates (centered at origin)
    octa.num_vertices = 6;
    octa.vertices[0] = (Point3D){  scale,  0,  0};  // +x
    octa.vertices[1] = (Point3D){ -scale,  0,  0};  // -x
    octa.vertices[2] = (Point3D){  0,  scale,  0};  // +y
    octa.vertices[3] = (Point3D){  0, -scale,  0};  // -y
    octa.vertices[4] = (Point3D){  0,  0,  scale};  // +z
    octa.vertices[5] = (Point3D){  0,  0, -scale};  // -z
    
    // Face definitions (CCW order when viewed from outside)
    octa.num_faces = 8;
    
    // Face 0: +x, +y, +z
    octa.faces[0].num_vertices = 3;
    octa.faces[0].vertices[0] = 0;
    octa.faces[0].vertices[1] = 2;
    octa.faces[0].vertices[2] = 4;
    
    // Face 1: +x, +z, -y
    octa.faces[1].num_vertices = 3;
    octa.faces[1].vertices[0] = 0;
    octa.faces[1].vertices[1] = 4;
    octa.faces[1].vertices[2] = 3;
    
    // Face 2: +x, -y, -z
    octa.faces[2].num_vertices = 3;
    octa.faces[2].vertices[0] = 0;
    octa.faces[2].vertices[1] = 3;
    octa.faces[2].vertices[2] = 5;
    
    // Face 3: +x, -z, +y
    octa.faces[3].num_vertices = 3;
    octa.faces[3].vertices[0] = 0;
    octa.faces[3].vertices[1] = 5;
    octa.faces[3].vertices[2] = 2;
    
    // Face 4: -x, +y, +z
    octa.faces[4].num_vertices = 3;
    octa.faces[4].vertices[0] = 1;
    octa.faces[4].vertices[1] = 4;
    octa.faces[4].vertices[2] = 2;
    
    // Face 5: -x, +z, -y
    octa.faces[5].num_vertices = 3;
    octa.faces[5].vertices[0] = 1;
    octa.faces[5].vertices[1] = 3;
    octa.faces[5].vertices[2] = 4;
    
    // Face 6: -x, -y, -z
    octa.faces[6].num_vertices = 3;
    octa.faces[6].vertices[0] = 1;
    octa.faces[6].vertices[1] = 5;
    octa.faces[6].vertices[2] = 3;
    
    // Face 7: -x, -z, +y
    octa.faces[7].num_vertices = 3;
    octa.faces[7].vertices[0] = 1;
    octa.faces[7].vertices[1] = 2;
    octa.faces[7].vertices[2] = 5;
    
    // Generate edges
    generate_edges_from_faces(&octa);
    
    // Calculate other properties
    calculate_solid_properties(&octa);
    
    return octa;
}

// Create a dodecahedron with the given edge length
PlatonicSolid create_dodecahedron(double edge_length) {
    PlatonicSolid dodeca;
    strcpy(dodeca.name, "Dodecahedron");
    dodeca.type = DODECAHEDRON;
    dodeca.edge_length = edge_length;
    
    // Basic properties
    dodeca.vertices_per_face = 5;  // Pentagonal faces
    dodeca.faces_per_vertex = 3;   // 3 faces meet at each vertex
    dodeca.dihedral_angle = acos(-sqrt(5)/5);  // ~116.57 degrees
    
    // For a dodecahedron, we'll use the golden ratio (phi)
    double phi = PHI;
    
    // Calculate scale factor for desired edge length
    double unit_edge = 2.0 / (1.0 + sqrt(5.0));  // Edge length in unit dodecahedron
    double scale = edge_length / unit_edge;
    
    // Vertex coordinates (centered at origin)
    dodeca.num_vertices = 20;
    
    // Vertices are based on three mutually orthogonal golden rectangles
    double a = scale;
    double b = scale * phi;
    
    // Cube vertices
    dodeca.vertices[0] = (Point3D){ a,  a,  a};
    dodeca.vertices[1] = (Point3D){ a,  a, -a};
    dodeca.vertices[2] = (Point3D){ a, -a,  a};
    dodeca.vertices[3] = (Point3D){ a, -a, -a};
    dodeca.vertices[4] = (Point3D){-a,  a,  a};
    dodeca.vertices[5] = (Point3D){-a,  a, -a};
    dodeca.vertices[6] = (Point3D){-a, -a,  a};
    dodeca.vertices[7] = (Point3D){-a, -a, -a};
    
    // Golden rectangle vertices (xy plane)
    dodeca.vertices[8]  = (Point3D){ 0,  b,  a/phi};
    dodeca.vertices[9]  = (Point3D){ 0,  b, -a/phi};
    dodeca.vertices[10] = (Point3D){ 0, -b,  a/phi};
    dodeca.vertices[11] = (Point3D){ 0, -b, -a/phi};
    
    // Golden rectangle vertices (xz plane)
    dodeca.vertices[12] = (Point3D){ b,  a/phi,  0};
    dodeca.vertices[13] = (Point3D){ b, -a/phi,  0};
    dodeca.vertices[14] = (Point3D){-b,  a/phi,  0};
    dodeca.vertices[15] = (Point3D){-b, -a/phi,  0};
    
    // Golden rectangle vertices (yz plane)
    dodeca.vertices[16] = (Point3D){ a/phi,  0,  b};
    dodeca.vertices[17] = (Point3D){-a/phi,  0,  b};
    dodeca.vertices[18] = (Point3D){ a/phi,  0, -b};
    dodeca.vertices[19] = (Point3D){-a/phi,  0, -b};
    
    // Face definitions (12 pentagonal faces)
    dodeca.num_faces = 12;
    
    // Face 0
    dodeca.faces[0].num_vertices = 5;
    dodeca.faces[0].vertices[0] = 0;
    dodeca.faces[0].vertices[1] = 8;
    dodeca.faces[0].vertices[2] = 4;
    dodeca.faces[0].vertices[3] = 17;
    dodeca.faces[0].vertices[4] = 16;
    
    // Face 1
    dodeca.faces[1].num_vertices = 5;
    dodeca.faces[1].vertices[0] = 0;
    dodeca.faces[1].vertices[1] = 16;
    dodeca.faces[1].vertices[2] = 2;
    dodeca.faces[1].vertices[3] = 10;
    dodeca.faces[1].vertices[4] = 8;
    
    // Face 2
    dodeca.faces[2].num_vertices = 5;
    dodeca.faces[2].vertices[0] = 0;
    dodeca.faces[2].vertices[1] = 12;
    dodeca.faces[2].vertices[2] = 1;
    dodeca.faces[2].vertices[3] = 9;
    dodeca.faces[2].vertices[4] = 8;
    
    // Face 3
    dodeca.faces[3].num_vertices = 5;
    dodeca.faces[3].vertices[0] = 0;
    dodeca.faces[3].vertices[1] = 16;
    dodeca.faces[3].vertices[2] = 17;
    dodeca.faces[3].vertices[3] = 6;
    dodeca.faces[3].vertices[4] = 2;
    
    // Face 4
    dodeca.faces[4].num_vertices = 5;
    dodeca.faces[4].vertices[0] = 1;
    dodeca.faces[4].vertices[1] = 12;
    dodeca.faces[4].vertices[2] = 13;
    dodeca.faces[4].vertices[3] = 3;
    dodeca.faces[4].vertices[4] = 18;
    
    // Face 5
    dodeca.faces[5].num_vertices = 5;
    dodeca.faces[5].vertices[0] = 1;
    dodeca.faces[5].vertices[1] = 18;
    dodeca.faces[5].vertices[2] = 19;
    dodeca.faces[5].vertices[3] = 5;
    dodeca.faces[5].vertices[4] = 9;
    
    // Face 6
    dodeca.faces[6].num_vertices = 5;
    dodeca.faces[6].vertices[0] = 2;
    dodeca.faces[6].vertices[1] = 6;
    dodeca.faces[6].vertices[2] = 10;
    dodeca.faces[6].vertices[3] = 11;
    dodeca.faces[6].vertices[4] = 3;
    
    // Face 7
    dodeca.faces[7].num_vertices = 5;
    dodeca.faces[7].vertices[0] = 2;
    dodeca.faces[7].vertices[1] = 13;
    dodeca.faces[7].vertices[2] = 12;
    dodeca.faces[7].vertices[3] = 0;
    dodeca.faces[7].vertices[4] = 16;
    
    // Face 8
    dodeca.faces[8].num_vertices = 5;
    dodeca.faces[8].vertices[0] = 3;
    dodeca.faces[8].vertices[1] = 11;
    dodeca.faces[8].vertices[2] = 7;
    dodeca.faces[8].vertices[3] = 19;
    dodeca.faces[8].vertices[4] = 18;
    
    // Face 9
    dodeca.faces[9].num_vertices = 5;
    dodeca.faces[9].vertices[0] = 3;
    dodeca.faces[9].vertices[1] = 13;
    dodeca.faces[9].vertices[2] = 2;
    dodeca.faces[9].vertices[3] = 6;
    dodeca.faces[9].vertices[4] = 15;
    
    // Face 10
    dodeca.faces[10].num_vertices = 5;
    dodeca.faces[10].vertices[0] = 4;
    dodeca.faces[10].vertices[1] = 8;
    dodeca.faces[10].vertices[2] = 9;
    dodeca.faces[10].vertices[3] = 5;
    dodeca.faces[10].vertices[4] = 14;
    
    // Face 11
    dodeca.faces[11].num_vertices = 5;
    dodeca.faces[11].vertices[0] = 4;
    dodeca.faces[11].vertices[1] = 14;
    dodeca.faces[11].vertices[2] = 15;
    dodeca.faces[11].vertices[3] = 6;
    dodeca.faces[11].vertices[4] = 17;
    
    // Generate edges
    generate_edges_from_faces(&dodeca);
    
    // Calculate other properties
    calculate_solid_properties(&dodeca);
    
    return dodeca;
}

// Create an icosahedron with the given edge length
PlatonicSolid create_icosahedron(double edge_length) {
    PlatonicSolid icosa;
    strcpy(icosa.name, "Icosahedron");
    icosa.type = ICOSAHEDRON;
    icosa.edge_length = edge_length;
    
    // Basic properties
    icosa.vertices_per_face = 3;  // Triangular faces
    icosa.faces_per_vertex = 5;   // 5 faces meet at each vertex
    icosa.dihedral_angle = acos(-sqrt(5)/3);  // ~138.19 degrees
    
    // Calculate scale factor for desired edge length
    double phi = PHI;  // Golden ratio
    double unit_edge = 2.0 / sqrt(phi + 2.0);  // Edge length in unit icosahedron
    double scale = edge_length / unit_edge;
    
    // Vertex coordinates (centered at origin)
    icosa.num_vertices = 12;
    
    // Vertices are based on three orthogonal golden rectangles
    double a = scale;
    double b = scale * phi;
    
    // Vertices on the coordinate axes
    icosa.vertices[0]  = (Point3D){ 0,  a,  b};
    icosa.vertices[1]  = (Point3D){ 0,  a, -b};
    icosa.vertices[2]  = (Point3D){ 0, -a,  b};
    icosa.vertices[3]  = (Point3D){ 0, -a, -b};
    icosa.vertices[4]  = (Point3D){ a,  b,  0};
    icosa.vertices[5]  = (Point3D){ a, -b,  0};
    icosa.vertices[6]  = (Point3D){-a,  b,  0};
    icosa.vertices[7]  = (Point3D){-a, -b,  0};
    icosa.vertices[8]  = (Point3D){ b,  0,  a};
    icosa.vertices[9]  = (Point3D){-b,  0,  a};
    icosa.vertices[10] = (Point3D){ b,  0, -a};
    icosa.vertices[11] = (Point3D){-b,  0, -a};
    
    // Face definitions (20 triangular faces)
    icosa.num_faces = 20;
    
    // 5 faces around vertex 0
    icosa.faces[0].num_vertices = 3;
    icosa.faces[0].vertices[0] = 0;
    icosa.faces[0].vertices[1] = 2;
    icosa.faces[0].vertices[2] = 8;
    
    icosa.faces[1].num_vertices = 3;
    icosa.faces[1].vertices[0] = 0;
    icosa.faces[1].vertices[1] = 8;
    icosa.faces[1].vertices[2] = 4;
    
    icosa.faces[2].num_vertices = 3;
    icosa.faces[2].vertices[0] = 0;
    icosa.faces[2].vertices[1] = 4;
    icosa.faces[2].vertices[2] = 6;
    
    icosa.faces[3].num_vertices = 3;
    icosa.faces[3].vertices[0] = 0;
    icosa.faces[3].vertices[1] = 6;
    icosa.faces[3].vertices[2] = 9;
    
    icosa.faces[4].num_vertices = 3;
    icosa.faces[4].vertices[0] = 0;
    icosa.faces[4].vertices[1] = 9;
    icosa.faces[4].vertices[2] = 2;
    
    // 5 faces around vertex 3
    icosa.faces[5].num_vertices = 3;
    icosa.faces[5].vertices[0] = 3;
    icosa.faces[5].vertices[1] = 11;
    icosa.faces[5].vertices[2] = 7;
    
    icosa.faces[6].num_vertices = 3;
    icosa.faces[6].vertices[0] = 3;
    icosa.faces[6].vertices[1] = 7;
    icosa.faces[6].vertices[2] = 5;
    
    icosa.faces[7].num_vertices = 3;
    icosa.faces[7].vertices[0] = 3;
    icosa.faces[7].vertices[1] = 5;
    icosa.faces[7].vertices[2] = 10;
    
    icosa.faces[8].num_vertices = 3;
    icosa.faces[8].vertices[0] = 3;
    icosa.faces[8].vertices[1] = 10;
    icosa.faces[8].vertices[2] = 1;
    
    icosa.faces[9].num_vertices = 3;
    icosa.faces[9].vertices[0] = 3;
    icosa.faces[9].vertices[1] = 1;
    icosa.faces[9].vertices[2] = 11;
    
    // 5 faces around equator
    icosa.faces[10].num_vertices = 3;
    icosa.faces[10].vertices[0] = 2;
    icosa.faces[10].vertices[1] = 9;
    icosa.faces[10].vertices[2] = 7;
    
    icosa.faces[11].num_vertices = 3;
    icosa.faces[11].vertices[0] = 2;
    icosa.faces[11].vertices[1] = 7;
    icosa.faces[11].vertices[2] = 5;
    
    icosa.faces[12].num_vertices = 3;
    icosa.faces[12].vertices[0] = 2;
    icosa.faces[12].vertices[1] = 5;
    icosa.faces[12].vertices[2] = 8;
    
    icosa.faces[13].num_vertices = 3;
    icosa.faces[13].vertices[0] = 8;
    icosa.faces[13].vertices[1] = 5;
    icosa.faces[13].vertices[2] = 10;
    
    icosa.faces[14].num_vertices = 3;
    icosa.faces[14].vertices[0] = 8;
    icosa.faces[14].vertices[1] = 10;
    icosa.faces[14].vertices[2] = 4;
    
    icosa.faces[15].num_vertices = 3;
    icosa.faces[15].vertices[0] = 4;
    icosa.faces[15].vertices[1] = 10;
    icosa.faces[15].vertices[2] = 1;
    
    icosa.faces[16].num_vertices = 3;
    icosa.faces[16].vertices[0] = 4;
    icosa.faces[16].vertices[1] = 1;
    icosa.faces[16].vertices[2] = 6;
    
    icosa.faces[17].num_vertices = 3;
    icosa.faces[17].vertices[0] = 6;
    icosa.faces[17].vertices[1] = 1;
    icosa.faces[17].vertices[2] = 11;
    
    icosa.faces[18].num_vertices = 3;
    icosa.faces[18].vertices[0] = 6;
    icosa.faces[18].vertices[1] = 11;
    icosa.faces[18].vertices[2] = 9;
    
    icosa.faces[19].num_vertices = 3;
    icosa.faces[19].vertices[0] = 9;
    icosa.faces[19].vertices[1] = 11;
    icosa.faces[19].vertices[2] = 7;
    
    // Generate edges
    generate_edges_from_faces(&icosa);
    
    // Calculate other properties
    calculate_solid_properties(&icosa);
    
    return icosa;
}

/* Main function with examples */
int main() {
    // Create all five Platonic solids with an edge length of 2.0
    double edge_length = 2.0;
    
    PlatonicSolid tetrahedron = create_tetrahedron(edge_length);
    PlatonicSolid cube = create_cube(edge_length);
    PlatonicSolid octahedron = create_octahedron(edge_length);
    PlatonicSolid dodecahedron = create_dodecahedron(edge_length);
    PlatonicSolid icosahedron = create_icosahedron(edge_length);
    
    // Print properties of each solid
    printf("PLATONIC SOLIDS WITH EDGE LENGTH %.2f\n", edge_length);
    printf("=======================================\n\n");
    
    print_solid_properties(tetrahedron);
    print_solid_properties(cube);
    print_solid_properties(octahedron);
    print_solid_properties(dodecahedron);
    print_solid_properties(icosahedron);
    
    // Verify Euler's formula for all solids: V - E + F = 2
    printf("Verification of Euler's formula (V - E + F = 2):\n");
    printf("Tetrahedron: %d - %d + %d = %d\n", 
           tetrahedron.num_vertices, tetrahedron.num_edges, tetrahedron.num_faces,
           tetrahedron.num_vertices - tetrahedron.num_edges + tetrahedron.num_faces);
    
    printf("Cube: %d - %d + %d = %d\n", 
           cube.num_vertices, cube.num_edges, cube.num_faces,
           cube.num_vertices - cube.num_edges + cube.num_faces);
    
    printf("Octahedron: %d - %d + %d = %d\n", 
           octahedron.num_vertices, octahedron.num_edges, octahedron.num_faces,
           octahedron.num_vertices - octahedron.num_edges + octahedron.num_faces);
    
    printf("Dodecahedron: %d - %d + %d = %d\n", 
           dodecahedron.num_vertices, dodecahedron.num_edges, dodecahedron.num_faces,
           dodecahedron.num_vertices - dodecahedron.num_edges + dodecahedron.num_faces);
    
    printf("Icosahedron: %d - %d + %d = %d\n", 
           icosahedron.num_vertices, icosahedron.num_edges, icosahedron.num_faces,
           icosahedron.num_vertices - icosahedron.num_edges + icosahedron.num_faces);
    
    return 0;
}