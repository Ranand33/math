/*
 * GEOMETRIC CONSTRUCTS IN C
 * -------------------------
 * This code defines and implements various geometric constructs:
 * 
 * - Points and Vectors (2D, 3D, nD)
 * - Lines, Rays, and Segments
 * - Polygons and Polyhedra
 * - Curves and Parametric Surfaces
 * - Circles, Spheres, and Ellipsoids
 * - Basic Manifold representation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#define MAX_DIMENSIONS 10
#define MAX_VERTICES 1000
#define MAX_FACES 1000
#define MAX_EDGES 3000
#define EPSILON 1e-10
#define PI 3.14159265358979323846

/*
 * PART 1: POINTS, VECTORS, AND COORDINATES
 */

// n-dimensional point/vector
typedef struct {
    int dim;                    // Number of dimensions
    double coords[MAX_DIMENSIONS]; // Coordinates
} Point;

// 2D point (for convenience)
typedef struct {
    double x;
    double y;
} Point2D;

// 3D point (for convenience)
typedef struct {
    double x;
    double y;
    double z;
} Point3D;

// Vector operations
double vector_magnitude(Point v);
double dot_product(Point v1, Point v2);
Point cross_product(Point3D v1, Point3D v2);
double distance(Point p1, Point p2);
Point vector_normalize(Point v);

/*
 * PART 2: LINES, RAYS, AND SEGMENTS
 */

// Line representation (parametric form: p + t*v)
typedef struct {
    Point point;       // Point on the line
    Point direction;   // Direction vector
} Line;

// Line segment
typedef struct {
    Point start;
    Point end;
} LineSegment;

// Ray (semi-infinite line)
typedef struct {
    Point origin;
    Point direction;
} Ray;

/*
 * PART 3: POLYGONS AND POLYHEDRA
 */

// Triangle (fundamental polygon)
typedef struct {
    Point vertices[3];
} Triangle;

// Polygon
typedef struct {
    int num_vertices;
    Point vertices[MAX_VERTICES];
} Polygon;

// Polyhedron
typedef struct {
    int num_vertices;
    int num_faces;
    int num_edges;
    Point3D vertices[MAX_VERTICES];
    
    // Each face is defined as indices to vertices
    struct {
        int num_vertices;
        int vertex_indices[MAX_VERTICES];
    } faces[MAX_FACES];
    
    struct {
        int vertex_indices[2];
    } edges[MAX_EDGES];
} Polyhedron;

/*
 * PART 4: CURVES AND PARAMETRIC SURFACES
 */

// Function pointer type for parametric curves
typedef Point (*ParametricFunction)(double t, void* params);

// Parametric curve
typedef struct {
    ParametricFunction function;
    void* params;
    double t_min;
    double t_max;
} ParametricCurve;

// Function pointer for parametric surfaces
typedef Point3D (*ParametricSurfaceFunction)(double u, double v, void* params);

// Parametric surface
typedef struct {
    ParametricSurfaceFunction function;
    void* params;
    double u_min, u_max;
    double v_min, v_max;
} ParametricSurface;

/*
 * PART 5: CIRCLES, SPHERES, AND QUADRICS
 */

// Circle
typedef struct {
    Point2D center;
    double radius;
} Circle;

// Sphere
typedef struct {
    Point3D center;
    double radius;
} Sphere;

// Ellipsoid
typedef struct {
    Point3D center;
    double semi_axes[3];    // a, b, c
    // Optional: rotation matrix (not included for simplicity)
} Ellipsoid;

// General quadric surface (ax² + by² + cz² + 2fyz + 2gxz + 2hxy + 2px + 2qy + 2rz + d = 0)
typedef struct {
    double coefficients[10]; // [a, b, c, f, g, h, p, q, r, d]
} QuadricSurface;

/*
 * PART 6: MANIFOLDS AND DIFFERENTIAL GEOMETRY
 */

// Tangent vector
typedef struct {
    Point base_point;   // Point on the manifold
    Point vector;       // Tangent vector at the point
} TangentVector;

// Simple representation of a differential manifold
typedef struct {
    // Function that maps from parameter space to manifold
    // (chart map or parameterization)
    ParametricSurfaceFunction parameterization;
    
    // Function that computes the metric tensor at a point
    void (*metric_tensor)(double u, double v, double g[2][2], void* params);
    
    void* params;
    
    // Chart domain
    double u_min, u_max;
    double v_min, v_max;
} Manifold2D;

/*******************************************/
/* IMPLEMENTATION OF FUNCTIONS AND METHODS */
/*******************************************/

/*
 * POINT AND VECTOR OPERATIONS
 */

// Create an n-dimensional point
Point create_point(int dim, double coords[]) {
    Point p;
    p.dim = dim;
    
    for (int i = 0; i < dim && i < MAX_DIMENSIONS; i++) {
        p.coords[i] = coords[i];
    }
    
    return p;
}

// Create a 2D point
Point2D create_point2D(double x, double y) {
    Point2D p = {x, y};
    return p;
}

// Create a 3D point
Point3D create_point3D(double x, double y, double z) {
    Point3D p = {x, y, z};
    return p;
}

// Convert 2D point to general point
Point point2D_to_point(Point2D p) {
    Point result;
    result.dim = 2;
    result.coords[0] = p.x;
    result.coords[1] = p.y;
    return result;
}

// Convert 3D point to general point
Point point3D_to_point(Point3D p) {
    Point result;
    result.dim = 3;
    result.coords[0] = p.x;
    result.coords[1] = p.y;
    result.coords[2] = p.z;
    return result;
}

// Vector addition
Point vector_add(Point v1, Point v2) {
    if (v1.dim != v2.dim) {
        fprintf(stderr, "Error: Cannot add vectors of different dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    Point result;
    result.dim = v1.dim;
    
    for (int i = 0; i < result.dim; i++) {
        result.coords[i] = v1.coords[i] + v2.coords[i];
    }
    
    return result;
}

// Vector subtraction
Point vector_subtract(Point v1, Point v2) {
    if (v1.dim != v2.dim) {
        fprintf(stderr, "Error: Cannot subtract vectors of different dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    Point result;
    result.dim = v1.dim;
    
    for (int i = 0; i < result.dim; i++) {
        result.coords[i] = v1.coords[i] - v2.coords[i];
    }
    
    return result;
}

// Vector scaling
Point vector_scale(Point v, double scalar) {
    Point result;
    result.dim = v.dim;
    
    for (int i = 0; i < result.dim; i++) {
        result.coords[i] = v.coords[i] * scalar;
    }
    
    return result;
}

// Vector magnitude/length
double vector_magnitude(Point v) {
    double sum_squares = 0.0;
    
    for (int i = 0; i < v.dim; i++) {
        sum_squares += v.coords[i] * v.coords[i];
    }
    
    return sqrt(sum_squares);
}

// Dot product
double dot_product(Point v1, Point v2) {
    if (v1.dim != v2.dim) {
        fprintf(stderr, "Error: Cannot compute dot product of vectors with different dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    double result = 0.0;
    
    for (int i = 0; i < v1.dim; i++) {
        result += v1.coords[i] * v2.coords[i];
    }
    
    return result;
}

// Cross product (only for 3D vectors)
Point cross_product3D(Point3D v1, Point3D v2) {
    Point result;
    result.dim = 3;
    
    result.coords[0] = v1.y * v2.z - v1.z * v2.y;
    result.coords[1] = v1.z * v2.x - v1.x * v2.z;
    result.coords[2] = v1.x * v2.y - v1.y * v2.x;
    
    return result;
}

// Distance between two points
double distance(Point p1, Point p2) {
    if (p1.dim != p2.dim) {
        fprintf(stderr, "Error: Cannot compute distance between points of different dimensions\n");
        exit(EXIT_FAILURE);
    }
    
    double sum_squares = 0.0;
    
    for (int i = 0; i < p1.dim; i++) {
        double diff = p1.coords[i] - p2.coords[i];
        sum_squares += diff * diff;
    }
    
    return sqrt(sum_squares);
}

// Normalize a vector
Point vector_normalize(Point v) {
    double mag = vector_magnitude(v);
    
    if (mag < EPSILON) {
        fprintf(stderr, "Warning: Normalizing a zero or near-zero vector\n");
        return v;
    }
    
    return vector_scale(v, 1.0 / mag);
}

/*
 * LINE OPERATIONS
 */

// Create a line from a point and direction
Line create_line(Point point, Point direction) {
    Line line;
    line.point = point;
    line.direction = direction;
    return line;
}

// Create a line from two points
Line create_line_from_points(Point p1, Point p2) {
    Line line;
    line.point = p1;
    line.direction = vector_subtract(p2, p1);
    return line;
}

// Create a line segment
LineSegment create_line_segment(Point start, Point end) {
    LineSegment segment;
    segment.start = start;
    segment.end = end;
    return segment;
}

// Length of a line segment
double line_segment_length(LineSegment segment) {
    return distance(segment.start, segment.end);
}

// Get a point on a line at parameter t
Point point_on_line(Line line, double t) {
    Point scaled_direction = vector_scale(line.direction, t);
    return vector_add(line.point, scaled_direction);
}

// Distance from a point to a line
double distance_point_to_line(Point point, Line line) {
    if (point.dim != line.point.dim) {
        fprintf(stderr, "Error: Point and line must have the same dimension\n");
        exit(EXIT_FAILURE);
    }
    
    // Vector from line point to the given point
    Point v = vector_subtract(point, line.point);
    
    // Normalize the line direction
    Point unit_direction = vector_normalize(line.direction);
    
    // Project v onto the line direction
    double projection = dot_product(v, unit_direction);
    Point projection_vector = vector_scale(unit_direction, projection);
    
    // Perpendicular component
    Point perpendicular = vector_subtract(v, projection_vector);
    
    return vector_magnitude(perpendicular);
}

/*
 * POLYGON OPERATIONS
 */

// Create a triangle
Triangle create_triangle(Point v1, Point v2, Point v3) {
    Triangle t;
    t.vertices[0] = v1;
    t.vertices[1] = v2;
    t.vertices[2] = v3;
    return t;
}

// Calculate triangle area
double triangle_area(Triangle t) {
    // Only works for 2D and 3D triangles
    if (t.vertices[0].dim < 2 || t.vertices[0].dim > 3) {
        fprintf(stderr, "Error: Triangle area calculation only supports 2D and 3D\n");
        exit(EXIT_FAILURE);
    }
    
    if (t.vertices[0].dim == 2) {
        // 2D case: use cross product magnitude / 2
        double x1 = t.vertices[0].coords[0];
        double y1 = t.vertices[0].coords[1];
        double x2 = t.vertices[1].coords[0];
        double y2 = t.vertices[1].coords[1];
        double x3 = t.vertices[2].coords[0];
        double y3 = t.vertices[2].coords[1];
        
        return 0.5 * fabs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)));
    } else {
        // 3D case: half the magnitude of cross product of two sides
        Point3D v1 = {t.vertices[0].coords[0], t.vertices[0].coords[1], t.vertices[0].coords[2]};
        Point3D v2 = {t.vertices[1].coords[0], t.vertices[1].coords[1], t.vertices[1].coords[2]};
        Point3D v3 = {t.vertices[2].coords[0], t.vertices[2].coords[1], t.vertices[2].coords[2]};
        
        Point3D side1 = {v2.x - v1.x, v2.y - v1.y, v2.z - v1.z};
        Point3D side2 = {v3.x - v1.x, v3.y - v1.y, v3.z - v1.z};
        
        Point cross = cross_product3D(side1, side2);
        return 0.5 * vector_magnitude(cross);
    }
}

// Create a polygon
Polygon create_polygon(int num_vertices, Point vertices[]) {
    Polygon poly;
    
    if (num_vertices > MAX_VERTICES) {
        fprintf(stderr, "Error: Number of vertices exceeds maximum\n");
        exit(EXIT_FAILURE);
    }
    
    poly.num_vertices = num_vertices;
    
    for (int i = 0; i < num_vertices; i++) {
        poly.vertices[i] = vertices[i];
    }
    
    return poly;
}

// Calculate polygon area (only for 2D polygons)
double polygon_area(Polygon poly) {
    // Only works for 2D polygons
    if (poly.vertices[0].dim != 2) {
        fprintf(stderr, "Error: Polygon area calculation only supports 2D polygons\n");
        exit(EXIT_FAILURE);
    }
    
    double area = 0.0;
    
    for (int i = 0; i < poly.num_vertices; i++) {
        int j = (i + 1) % poly.num_vertices;
        area += poly.vertices[i].coords[0] * poly.vertices[j].coords[1];
        area -= poly.vertices[j].coords[0] * poly.vertices[i].coords[1];
    }
    
    return 0.5 * fabs(area);
}

// Check if a 2D point is inside a 2D polygon (ray casting algorithm)
bool point_in_polygon(Point2D point, Polygon poly) {
    // Only works for 2D
    if (poly.vertices[0].dim != 2) {
        fprintf(stderr, "Error: Point-in-polygon test only supports 2D\n");
        exit(EXIT_FAILURE);
    }
    
    bool inside = false;
    
    for (int i = 0, j = poly.num_vertices - 1; i < poly.num_vertices; j = i++) {
        double xi = poly.vertices[i].coords[0];
        double yi = poly.vertices[i].coords[1];
        double xj = poly.vertices[j].coords[0];
        double yj = poly.vertices[j].coords[1];
        
        bool intersect = ((yi > point.y) != (yj > point.y)) &&
                         (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
        
        if (intersect) {
            inside = !inside;
        }
    }
    
    return inside;
}

/*
 * CIRCLE AND SPHERE OPERATIONS
 */

// Create a circle
Circle create_circle(Point2D center, double radius) {
    Circle circle;
    circle.center = center;
    circle.radius = radius;
    return circle;
}

// Create a sphere
Sphere create_sphere(Point3D center, double radius) {
    Sphere sphere;
    sphere.center = center;
    sphere.radius = radius;
    return sphere;
}

// Circle area
double circle_area(Circle circle) {
    return PI * circle.radius * circle.radius;
}

// Circle circumference
double circle_circumference(Circle circle) {
    return 2.0 * PI * circle.radius;
}

// Sphere volume
double sphere_volume(Sphere sphere) {
    return (4.0 / 3.0) * PI * pow(sphere.radius, 3);
}

// Sphere surface area
double sphere_surface_area(Sphere sphere) {
    return 4.0 * PI * sphere.radius * sphere.radius;
}

// Check if a point is inside a circle
bool point_in_circle(Point2D point, Circle circle) {
    double dx = point.x - circle.center.x;
    double dy = point.y - circle.center.y;
    double distance_squared = dx * dx + dy * dy;
    
    return distance_squared <= circle.radius * circle.radius;
}

// Check if a point is inside a sphere
bool point_in_sphere(Point3D point, Sphere sphere) {
    double dx = point.x - sphere.center.x;
    double dy = point.y - sphere.center.y;
    double dz = point.z - sphere.center.z;
    double distance_squared = dx * dx + dy * dy + dz * dz;
    
    return distance_squared <= sphere.radius * sphere.radius;
}

/*
 * PARAMETRIC CURVE AND SURFACE OPERATIONS
 */

// Example of a parametric curve: circle
Point circle_curve(double t, void* params) {
    Circle* circle = (Circle*)params;
    Point p;
    p.dim = 2;
    p.coords[0] = circle->center.x + circle->radius * cos(t);
    p.coords[1] = circle->center.y + circle->radius * sin(t);
    return p;
}

// Example of a parametric surface: sphere
Point3D sphere_surface(double u, double v, void* params) {
    Sphere* sphere = (Sphere*)params;
    Point3D p;
    p.x = sphere->center.x + sphere->radius * sin(u) * cos(v);
    p.y = sphere->center.y + sphere->radius * sin(u) * sin(v);
    p.z = sphere->center.z + sphere->radius * cos(u);
    return p;
}

// Create a parametric curve
ParametricCurve create_parametric_curve(ParametricFunction func, void* params, 
                                         double t_min, double t_max) {
    ParametricCurve curve;
    curve.function = func;
    curve.params = params;
    curve.t_min = t_min;
    curve.t_max = t_max;
    return curve;
}

// Create a parametric surface
ParametricSurface create_parametric_surface(ParametricSurfaceFunction func, void* params, 
                                            double u_min, double u_max,
                                            double v_min, double v_max) {
    ParametricSurface surface;
    surface.function = func;
    surface.params = params;
    surface.u_min = u_min;
    surface.u_max = u_max;
    surface.v_min = v_min;
    surface.v_max = v_max;
    return surface;
}

// Get a point on a parametric curve at parameter t
Point point_on_curve(ParametricCurve curve, double t) {
    // Ensure t is within bounds
    if (t < curve.t_min) t = curve.t_min;
    if (t > curve.t_max) t = curve.t_max;
    
    return curve.function(t, curve.params);
}

// Get a point on a parametric surface at parameters (u,v)
Point3D point_on_surface(ParametricSurface surface, double u, double v) {
    // Ensure u, v are within bounds
    if (u < surface.u_min) u = surface.u_min;
    if (u > surface.u_max) u = surface.u_max;
    if (v < surface.v_min) v = surface.v_min;
    if (v > surface.v_max) v = surface.v_max;
    
    return surface.function(u, v, surface.params);
}

/*
 * MANIFOLD OPERATIONS
 */

// Example of a metric tensor for a 2D manifold (sphere)
void sphere_metric(double u, double v, double g[2][2], void* params) {
    Sphere* sphere = (Sphere*)params;
    double r = sphere->radius;
    
    // Metric tensor for a sphere of radius r
    g[0][0] = r * r;
    g[0][1] = 0;
    g[1][0] = 0;
    g[1][1] = r * r * sin(u) * sin(u);
}

// Create a 2D manifold
Manifold2D create_manifold2D(ParametricSurfaceFunction param_func, 
                             void (*metric_func)(double, double, double[2][2], void*), 
                             void* params,
                             double u_min, double u_max,
                             double v_min, double v_max) {
    Manifold2D manifold;
    manifold.parameterization = param_func;
    manifold.metric_tensor = metric_func;
    manifold.params = params;
    manifold.u_min = u_min;
    manifold.u_max = u_max;
    manifold.v_min = v_min;
    manifold.v_max = v_max;
    return manifold;
}

// Calculate geodesic distance (very simplified)
double geodesic_distance(Manifold2D manifold, 
                         double u1, double v1, 
                         double u2, double v2) {
    // This is a very simplified approximation using Euclidean distance
    // in parameter space, weighted by the metric tensor
    
    // Get metric at midpoint
    double u_mid = (u1 + u2) / 2.0;
    double v_mid = (v1 + v2) / 2.0;
    double g[2][2];
    
    manifold.metric_tensor(u_mid, v_mid, g, manifold.params);
    
    // Parameter differences
    double du = u2 - u1;
    double dv = v2 - v1;
    
    // Approximate geodesic distance
    return sqrt(g[0][0] * du * du + 2 * g[0][1] * du * dv + g[1][1] * dv * dv);
}

/*
 * MAIN FUNCTION WITH EXAMPLES
 */

int main() {
    // Example 1: Create and operate on points and vectors
    double coords1[] = {1.0, 2.0, 3.0};
    double coords2[] = {4.0, 5.0, 6.0};
    
    Point p1 = create_point(3, coords1);
    Point p2 = create_point(3, coords2);
    
    Point3D p3d1 = create_point3D(1.0, 2.0, 3.0);
    
    printf("Point operations example:\n");
    printf("Distance between points: %.2f\n", distance(p1, p2));
    printf("Vector magnitude: %.2f\n", vector_magnitude(p1));
    
    // Example 2: Work with a line
    Line line = create_line_from_points(p1, p2);
    printf("\nLine operations example:\n");
    
    // Point at parameter t=0.5 (midpoint)
    Point mid_point = point_on_line(line, 0.5);
    printf("Midpoint coords: (%.2f, %.2f, %.2f)\n", 
           mid_point.coords[0], mid_point.coords[1], mid_point.coords[2]);
    
    // Example 3: Create and work with a triangle
    Triangle triangle = create_triangle(p1, p2, vector_scale(p1, 2.0));
    printf("\nTriangle example:\n");
    printf("Triangle area: %.2f\n", triangle_area(triangle));
    
    // Example 4: Create and work with a circle
    Point2D center = create_point2D(0.0, 0.0);
    Circle circle = create_circle(center, 5.0);
    
    printf("\nCircle example:\n");
    printf("Circle area: %.2f\n", circle_area(circle));
    printf("Circle circumference: %.2f\n", circle_circumference(circle));
    
    Point2D test_point = create_point2D(3.0, 4.0);
    printf("Is point (3,4) in circle? %s\n", 
           point_in_circle(test_point, circle) ? "Yes" : "No");
    
    // Example 5: Create and work with a sphere
    Point3D sphere_center = create_point3D(0.0, 0.0, 0.0);
    Sphere sphere = create_sphere(sphere_center, 5.0);
    
    printf("\nSphere example:\n");
    printf("Sphere volume: %.2f\n", sphere_volume(sphere));
    printf("Sphere surface area: %.2f\n", sphere_surface_area(sphere));
    
    // Example 6: Create a parametric curve (circle)
    ParametricCurve param_circle = create_parametric_curve(
        circle_curve, &circle, 0.0, 2.0 * PI);
    
    printf("\nParametric curve example:\n");
    Point curve_point = point_on_curve(param_circle, PI / 4.0);
    printf("Point on circle at t=π/4: (%.2f, %.2f)\n", 
           curve_point.coords[0], curve_point.coords[1]);
    
    // Example 7: Create a 2D manifold (sphere)
    Manifold2D sphere_manifold = create_manifold2D(
        sphere_surface, sphere_metric, &sphere,
        0.0, PI, 0.0, 2.0 * PI);
    
    printf("\nManifold example:\n");
    printf("Approximate geodesic distance on sphere: %.2f\n", 
           geodesic_distance(sphere_manifold, 0.0, 0.0, PI/2.0, 0.0));
    
    return 0;
}