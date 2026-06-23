#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>

constexpr float pi = 3.14159265358979f;

struct vec3 {
    float x=0, y=0, z=0;
          float& operator[](const int i)       { return i==0 ? x : (1==i ? y : z); }
    const float& operator[](const int i) const { return i==0 ? x : (1==i ? y : z); }
    vec3  operator*(const float v) const { return {x*v, y*v, z*v};       }
    float operator*(const vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    vec3  operator+(const vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    vec3  operator-(const vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    vec3  operator-()              const { return {-x, -y, -z};          }
    vec3  mult(const vec3& v)      const { return {x*v.x, y*v.y, z*v.z};  } // component-wise (Hadamard) product
    float max_component()          const { return std::max(x, std::max(y, z)); }
    float norm() const { return std::sqrt(x*x+y*y+z*z); }
    vec3 normalized() const { return (*this)*(1.f/norm()); }
};

vec3 cross(const vec3 &a, const vec3 &b) {
    return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

struct RNG { // xorshift64* — a tiny, fast, decent-quality stream of uniforms in [0,1)
    uint64_t s;
    float operator()() {
        s ^= s>>12; s ^= s<<25; s ^= s>>27;
        return float((s*0x2545F4914F6CDD1DULL) >> 40) * (1.f/16777216.f); // top 24 bits -> [0,1)
    }
};

uint64_t seed(uint64_t a, uint64_t b) { // splitmix64 hash, decorrelates the per-pixel/per-sample streams
    uint64_t z = a*0x9E3779B97F4A7C15ULL + b + 1;
    z = (z ^ (z>>30))*0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z>>27))*0x94D049BB133111EBULL;
    return z ^ (z>>31);
}

struct Material {                  // a small subset of a physically based "principled" surface
    vec3  albedo       = {0,0,0};  // base color (diffuse reflectance for dielectrics, F0 tint for metals)
    float roughness    = 1;        // 0 = mirror-smooth, 1 = fully rough
    float metalness    = 0;        // 0 = dielectric, 1 = conductor
    float ior          = 1.5;      // index of refraction (used by transparent dielectrics)
    float transparency = 0;        // 0 = opaque, 1 = perfect glass-like dielectric
    vec3  emission     = {0,0,0};  // self-emitted radiance (area/emissive surfaces)
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

constexpr Material      ivory = {{0.40, 0.40, 0.30}, 0.50, 0.0, 1.5, 0.0, {0,0,0}};
constexpr Material      glass = {{0.90, 0.95, 1.00}, 0.02, 0.0, 1.5, 1.0, {0,0,0}};
constexpr Material red_rubber = {{0.30, 0.10, 0.10}, 0.90, 0.0, 1.5, 0.0, {0,0,0}};
constexpr Material     mirror = {{0.95, 0.95, 0.95}, 0.03, 1.0, 1.5, 0.0, {0,0,0}};

constexpr Sphere spheres[] = {
    {{-3,    0,   -16}, 2,      ivory},
    {{-1.0, -1.5, -12}, 2,      glass},
    {{ 1.5, -0.5, -18}, 3, red_rubber},
    {{ 7,    5,   -18}, 4,     mirror}
};

constexpr vec3  lights[]     = {{-20, 20, 20}, {30, 50, -25}, {30, 20, 30}};
constexpr float light_power  = 10.f; // radiant intensity of each (delta) point light

vec3 reflect(const vec3 &I, const vec3 &N) {
    return I - N*2.f*(I*N);
}

bool refract(const vec3 &I, const vec3 &N, const float eta, vec3 &out) { // Snell's law; N faces the incident ray
    float cosi = -(I*N);
    float k = 1 - eta*eta*(1 - cosi*cosi);
    if (k<0) return false;                              // total internal reflection: nothing refracts
    out = I*eta + N*(eta*cosi - std::sqrt(k));
    return true;
}

std::tuple<bool,float> ray_sphere_intersect(const vec3 &orig, const vec3 &dir, const Sphere &s) { // ret value is a pair [intersection found, distance]
    vec3 L = s.center - orig;
    float tca = L*dir;
    float d2 = L*L - tca*tca;
    if (d2 > s.radius*s.radius) return {false, 0};
    float thc = std::sqrt(s.radius*s.radius - d2);
    float t0 = tca-thc, t1 = tca+thc;
    if (t0>.001) return {true, t0};  // offset the original point by .001 to avoid occlusion by the object itself
    if (t1>.001) return {true, t1};
    return {false, 0};
}

struct Triangle {
    vec3 v0, v1, v2;   // vertices
    vec3 n0, n1, n2;   // per-vertex normals (smooth shading); equal to the face normal for a flat triangle
    vec3 centroid() const { return (v0+v1+v2)*(1.f/3.f); }
};

std::tuple<bool,float,float,float> ray_triangle_intersect(const vec3 &orig, const vec3 &dir, const Triangle &tr) { // Moller–Trumbore; ret [hit, dist, barycentric u, v]
    vec3 e1 = tr.v1 - tr.v0, e2 = tr.v2 - tr.v0;
    vec3 pvec = cross(dir, e2);
    float det = e1*pvec;
    if (std::abs(det) < 1e-9f) return {false, 0, 0, 0};  // ray is parallel to the triangle
    float inv = 1.f/det;
    vec3 tvec = orig - tr.v0;
    float u = (tvec*pvec)*inv;
    if (u<0 || u>1) return {false, 0, 0, 0};
    vec3 qvec = cross(tvec, e1);
    float v = (dir*qvec)*inv;
    if (v<0 || u+v>1) return {false, 0, 0, 0};
    float t = (e2*qvec)*inv;
    if (t < 1e-3f) return {false, 0, 0, 0};
    return {true, t, u, v};
}

struct AABB {
    vec3 lo{ 1e30f,  1e30f,  1e30f};
    vec3 hi{-1e30f, -1e30f, -1e30f};
    void grow(const vec3 &p) { for (int k=0;k<3;k++) { lo[k]=std::min(lo[k],p[k]); hi[k]=std::max(hi[k],p[k]); } }
    void grow(const AABB &b) { grow(b.lo); grow(b.hi); }
    float area() const { vec3 d = hi-lo; return d.x<0 ? 0.f : 2.f*(d.x*d.y + d.y*d.z + d.z*d.x); }
    bool hit(const vec3 &orig, const vec3 &invd, float tmax) const { // ray/box slab test on [eps, tmax]
        float tmin = 1e-3f, tmx = tmax;
        for (int k=0;k<3;k++) {
            float t1=(lo[k]-orig[k])*invd[k], t2=(hi[k]-orig[k])*invd[k];
            if (t1>t2) std::swap(t1,t2);
            tmin = std::max(tmin,t1); tmx = std::min(tmx,t2);
        }
        return tmx >= tmin;
    }
};

struct BVHNode {
    AABB box;
    int left  = -1;        // left child index (right child is left+1); a leaf has left<0
    int start = 0, count = 0; // triangle range into BVH::index (leaves only)
};

// Bounding-volume hierarchy built with a binned Surface Area Heuristic, traversed front-to-back with an explicit stack.
struct BVH {
    static constexpr int BINS = 16;
    std::vector<Triangle> tris;
    std::vector<int>      index;
    std::vector<BVHNode>  nodes;

    void build() {
        index.resize(tris.size());
        for (size_t i=0;i<tris.size();i++) index[i] = int(i);
        nodes.clear();
        nodes.reserve(tris.empty() ? 1 : 2*tris.size()); // a BVH over N leaves has <= 2N-1 nodes -> no reallocation
        nodes.push_back({});
        if (!tris.empty()) build_node(0, 0, int(tris.size()));
    }

    void build_node(int node, int start, int count) {
        AABB box, cbox;                                  // box bounds the geometry, cbox bounds the centroids
        for (int i=start;i<start+count;i++) {
            const Triangle &t = tris[index[i]];
            box.grow(t.v0); box.grow(t.v1); box.grow(t.v2);
            cbox.grow(t.centroid());
        }
        nodes[node].box = box;

        int best_axis=-1, best_bin=-1;
        float best_cost = box.area()*count;              // SAH cost of making this node a leaf
        for (int axis=0; axis<3; axis++) {
            float lo = cbox.lo[axis], hi = cbox.hi[axis];
            if (hi-lo < 1e-8f) continue;                 // centroids coincide along this axis — nothing to split
            float scale = BINS/(hi-lo);
            AABB bin_box[BINS]; int bin_cnt[BINS] = {0};
            for (int i=start;i<start+count;i++) {
                const Triangle &t = tris[index[i]];
                int b = std::min(BINS-1, int((t.centroid()[axis]-lo)*scale));
                bin_cnt[b]++;
                bin_box[b].grow(t.v0); bin_box[b].grow(t.v1); bin_box[b].grow(t.v2);
            }
            float suf_area[BINS]; int suf_cnt[BINS];      // suffix sweep: area & count of bins [b..BINS)
            AABB acc; int cnt=0;
            for (int b=BINS-1;b>=0;b--) { acc.grow(bin_box[b]); cnt+=bin_cnt[b]; suf_area[b]=acc.area(); suf_cnt[b]=cnt; }
            AABB lacc; int lcnt=0;                        // prefix sweep: evaluate the SAH cost of every split plane
            for (int b=0;b<BINS-1;b++) {
                lacc.grow(bin_box[b]); lcnt+=bin_cnt[b];
                if (!lcnt || !suf_cnt[b+1]) continue;
                float cost = lacc.area()*lcnt + suf_area[b+1]*suf_cnt[b+1];
                if (cost < best_cost) { best_cost=cost; best_axis=axis; best_bin=b; }
            }
        }

        if (count<=4 || best_axis<0) {                   // leaf: small range, or no split beats keeping it whole
            nodes[node].left=-1; nodes[node].start=start; nodes[node].count=count;
            return;
        }

        int axis = best_axis;
        float lo = cbox.lo[axis], scale = BINS/(cbox.hi[axis]-cbox.lo[axis]);
        auto it = std::partition(index.begin()+start, index.begin()+start+count,
            [&](int id){ return std::min(BINS-1, int((tris[id].centroid()[axis]-lo)*scale)) <= best_bin; });
        int mid = int(it - index.begin());
        if (mid==start || mid==start+count) {            // one-sided split — fall back to a centroid median
            mid = start + count/2;
            std::nth_element(index.begin()+start, index.begin()+mid, index.begin()+start+count,
                [&](int x, int y){ return tris[x].centroid()[axis] < tris[y].centroid()[axis]; });
        }

        int lc = int(nodes.size());
        nodes.push_back({}); nodes.push_back({});
        nodes[node].left = lc; nodes[node].count = 0;
        build_node(lc,   start, mid-start);
        build_node(lc+1, mid,   start+count-mid);
    }

    std::tuple<bool,float,float,float,int> intersect(const vec3 &orig, const vec3 &dir) const { // ret [hit, dist, u, v, triangle index]
        if (nodes.empty()) return {false, 0, 0, 0, -1};
        vec3 invd{ 1.f/dir.x, 1.f/dir.y, 1.f/dir.z };
        float best_t=1e30f, bu=0, bv=0; int best=-1;
        int stack[64], sp=0; stack[sp++]=0;
        while (sp) {
            const BVHNode &nd = nodes[stack[--sp]];
            if (!nd.box.hit(orig, invd, best_t)) continue;
            if (nd.left<0) {
                for (int i=nd.start;i<nd.start+nd.count;i++) {
                    auto [h,t,u,v] = ray_triangle_intersect(orig, dir, tris[index[i]]);
                    if (h && t<best_t) { best_t=t; bu=u; bv=v; best=index[i]; }
                }
            } else { stack[sp++]=nd.left; stack[sp++]=nd.left+1; }
        }
        return { best>=0, best_t, bu, bv, best };
    }
};

BVH g_mesh;                                              // the triangle mesh, accelerated by its BVH
constexpr Material mesh_material = {{1.00, 0.78, 0.36}, 0.18, 1.0, 1.5, 0.0, {0,0,0}}; // polished gold

std::tuple<bool,vec3,vec3,Material> scene_intersect(const vec3 &orig, const vec3 &dir) {
    vec3 pt, N;
    Material material;

    float nearest_dist = 1e10;
    if (std::abs(dir.y)>.001) { // intersect the ray with the checkerboard, avoid division by zero
        float d = -(orig.y+4)/dir.y; // the checkerboard plane has equation y = -4
        vec3 p = orig + dir*d;
        if (d>.001 && d<nearest_dist && std::abs(p.x)<10 && p.z<-10 && p.z>-30) {
            nearest_dist = d;
            pt = p;
            N = {0,1,0};
            material = Material{};
            material.albedo = (int(.5*pt.x+1000) + int(.5*pt.z)) & 1 ? vec3{.3, .3, .3} : vec3{.3, .2, .1};
            material.roughness = 1;
        }
    }

    for (const Sphere &s : spheres) { // intersect the ray with all spheres
        auto [intersection, d] = ray_sphere_intersect(orig, dir, s);
        if (!intersection || d > nearest_dist) continue;
        nearest_dist = d;
        pt = orig + dir*nearest_dist;
        N = (pt - s.center).normalized();
        material = s.material;
    }

    auto [mhit, md, mu, mv, mi] = g_mesh.intersect(orig, dir); // intersect the triangle mesh through its BVH
    if (mhit && md < nearest_dist) {
        nearest_dist = md;
        pt = orig + dir*md;
        const Triangle &tr = g_mesh.tris[mi];
        vec3 n = tr.n0*(1-mu-mv) + tr.n1*mu + tr.n2*mv;  // barycentric-interpolated smooth normal
        N = (n.norm()>1e-6f ? n : cross(tr.v1-tr.v0, tr.v2-tr.v0)).normalized();
        material = mesh_material;
    }
    return { nearest_dist<1000, pt, N, material };
}

void onb(const vec3 &n, vec3 &t, vec3 &b) { // Duff et al. — branchless orthonormal basis around n
    float s = n.z<0.f ? -1.f : 1.f;
    float a = -1.f/(s+n.z);
    float d = n.x*n.y*a;
    t = {1.f + s*n.x*n.x*a, s*d, -s*n.x};
    b = {d, s + n.y*n.y*a, -n.y};
}

vec3 to_world(const vec3 &local, const vec3 &n) {
    vec3 t, b; onb(n, t, b);
    return (t*local.x + b*local.y + n*local.z).normalized();
}

vec3 cosine_sample(const vec3 &n, float u1, float u2) { // cosine-weighted hemisphere — importance samples the Lambert lobe
    float r = std::sqrt(u1), phi = 2*pi*u2;
    return to_world({r*std::cos(phi), r*std::sin(phi), std::sqrt(std::max(0.f, 1-u1))}, n);
}

vec3 ggx_sample_h(const vec3 &n, float a, float u1, float u2) { // sample a microfacet half-vector from the GGX distribution
    float phi = 2*pi*u1;
    float ct = std::sqrt((1-u2)/(1+(a*a-1)*u2));
    float st = std::sqrt(std::max(0.f, 1-ct*ct));
    return to_world({st*std::cos(phi), st*std::sin(phi), ct}, n);
}

float ggx_D(float NdotH, float a) { // Trowbridge–Reitz (GGX) normal distribution function
    float a2 = a*a;
    float d = NdotH*NdotH*(a2-1)+1;
    return a2 / (pi*d*d + 1e-7f);
}

float smith_G1(float NdotX, float a) { // Smith masking-shadowing term (one direction)
    return 2*NdotX / (NdotX + std::sqrt(a*a + (1-a*a)*NdotX*NdotX) + 1e-7f);
}

vec3 fresnel_schlick(float cosT, const vec3 &F0) { // Schlick approximation of the Fresnel reflectance
    return F0 + (vec3{1,1,1}-F0)*std::pow(std::max(0.f, 1-cosT), 5.f);
}

vec3 base_reflectance(const Material &m) { // F0: 4% for dielectrics, tinted albedo for metals
    return vec3{0.04f,0.04f,0.04f}*(1-m.metalness) + m.albedo*m.metalness;
}

vec3 brdf_eval(const Material &m, const vec3 &N, const vec3 &V, const vec3 &L) { // Cook–Torrance specular + Lambert diffuse
    float NdotL = N*L, NdotV = N*V;
    if (NdotL<=0 || NdotV<=0) return {0,0,0};
    vec3 H = (V+L).normalized();
    float NdotH = std::max(0.f, N*H), VdotH = std::max(0.f, V*H);
    float a = std::max(1e-3f, m.roughness*m.roughness);
    vec3  F = fresnel_schlick(VdotH, base_reflectance(m));
    float D = ggx_D(NdotH, a);
    float G = smith_G1(NdotV, a)*smith_G1(NdotL, a);
    vec3 specular = F*(D*G/(4*NdotV*NdotL + 1e-7f));
    vec3 diffuse  = m.albedo.mult((vec3{1,1,1}-F)*(1-m.metalness))*(1.f/pi); // (1-F) keeps the BRDF energy conserving
    return diffuse + specular;
}

float brdf_pdf(const Material &m, const vec3 &N, const vec3 &V, const vec3 &L, float p_spec) { // pdf of the mixed (diffuse+specular) sampler
    float NdotL = std::max(0.f, N*L);
    vec3 H = (V+L).normalized();
    float NdotH = std::max(0.f, N*H), VdotH = std::max(1e-4f, V*H);
    float a = std::max(1e-3f, m.roughness*m.roughness);
    float pdf_diff = NdotL/pi;
    float pdf_spec = ggx_D(NdotH, a)*NdotH/(4*VdotH); // Jacobian of the reflect(H) mapping is 1/(4*VdotH)
    return p_spec*pdf_spec + (1-p_spec)*pdf_diff;
}

vec3 sky(const vec3 &) {
    return {0.12f, 0.40f, 0.52f}; // uniform sky: both the background and the only area light source (soft ambient + AO)
}

bool occluded(const vec3 &orig, const vec3 &dir, float dist) {
    auto [hit, p, n, m] = scene_intersect(orig, dir);
    return hit && (p-orig).norm() < dist - 1e-3f;
}

// Solve the rendering equation by unbiased Monte-Carlo path tracing:
//   direct light via next-event estimation toward the point lights,
//   indirect light via one BSDF-importance-sampled bounce, terminated by Russian roulette.
vec3 path_trace(vec3 orig, vec3 dir, RNG &rng) {
    vec3 radiance{0,0,0}, throughput{1,1,1};
    for (int bounce=0; bounce<8; bounce++) {
        auto [hit, point, N, m] = scene_intersect(orig, dir);
        if (!hit) { radiance = radiance + throughput.mult(sky(dir)); break; }
        radiance = radiance + throughput.mult(m.emission);
        vec3 V = -dir;

        if (m.transparency>0) { // perfect dielectric: pick reflection or refraction stochastically by the Fresnel term
            vec3  Nf   = (dir*N<0) ? N : -N;             // geometric normal oriented against the incident ray
            float eta  = (dir*N<0) ? 1.f/m.ior : m.ior;  // ratio n_incident / n_transmitted
            float cosi = std::min(1.f, -(dir*Nf));
            float R0   = (m.ior-1)/(m.ior+1); R0 *= R0;
            float Fr   = R0 + (1-R0)*std::pow(1-cosi, 5.f);
            vec3 refr;
            if (!refract(dir, Nf, eta, refr) || rng()<Fr) dir = reflect(dir, Nf).normalized();
            else                                          dir = refr.normalized();
            orig = point;
            throughput = throughput.mult(m.albedo);      // optional glass tint
            continue;
        }

        // --- next-event estimation: connect to every point light directly (delta lights need no MIS) ---
        for (const vec3 &L : lights) {
            vec3  to_light = L - point;
            float dist = to_light.norm();
            vec3  Ldir = to_light*(1.f/dist);
            if (N*Ldir<=0 || occluded(point, Ldir, dist)) continue;
            radiance = radiance + throughput.mult(brdf_eval(m, N, V, Ldir))*(N*Ldir)*light_power;
        }

        // --- indirect: importance-sample one outgoing direction from the BSDF ---
        float p_spec = std::min(0.9f, std::max(0.25f, base_reflectance(m).max_component()));
        float a = std::max(1e-3f, m.roughness*m.roughness);
        vec3 wi = rng()<p_spec ? reflect(dir, ggx_sample_h(N, a, rng(), rng())).normalized()
                               : cosine_sample(N, rng(), rng());
        if (N*wi<=0) break;
        float pdf = brdf_pdf(m, N, V, wi, p_spec);
        if (pdf<=1e-6f) break;
        throughput = throughput.mult(brdf_eval(m, N, V, wi))*((N*wi)/pdf);
        orig = point; dir = wi;

        // --- Russian roulette: unbiased path termination once the throughput gets small ---
        if (bounce>2) {
            float q = std::min(0.95f, throughput.max_component());
            if (rng()>q) break;
            throughput = throughput*(1.f/q);
        }
    }
    return radiance;
}

std::vector<Triangle> load_obj(const std::string &path) { // minimal Wavefront OBJ reader: v / vn / f, polygons triangulated as fans
    std::ifstream in(path);
    std::vector<vec3> verts, norms;
    std::vector<Triangle> tris;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag; ss >> tag;
        if (tag=="v")  { vec3 p; ss>>p.x>>p.y>>p.z; verts.push_back(p); }
        else if (tag=="vn") { vec3 n; ss>>n.x>>n.y>>n.z; norms.push_back(n); }
        else if (tag=="f")  {
            std::vector<int> vi, ni;
            std::string tok;
            while (ss >> tok) {                          // each token is v, v/vt, v//vn or v/vt/vn
                int v=0, vn=0;
                size_t s1 = tok.find('/');
                if (s1==std::string::npos) v = std::stoi(tok);
                else {
                    v = std::stoi(tok.substr(0,s1));
                    size_t s2 = tok.find('/', s1+1);
                    if (s2!=std::string::npos && s2+1<tok.size()) vn = std::stoi(tok.substr(s2+1));
                }
                vi.push_back(v <0 ? int(verts.size())+v : v -1); // OBJ is 1-based; negative indices are relative
                ni.push_back(vn==0 ? -1 : (vn<0 ? int(norms.size())+vn : vn-1));
            }
            for (size_t k=1; k+1<vi.size(); k++) {       // fan-triangulate the face
                Triangle t;
                t.v0=verts[vi[0]]; t.v1=verts[vi[k]]; t.v2=verts[vi[k+1]];
                vec3 gn = cross(t.v1-t.v0, t.v2-t.v0).normalized();
                t.n0 = ni[0]  >=0 ? norms[ni[0]]  : gn;
                t.n1 = ni[k]  >=0 ? norms[ni[k]]  : gn;
                t.n2 = ni[k+1]>=0 ? norms[ni[k+1]]: gn;
                tris.push_back(t);
            }
        }
    }
    return tris;
}

std::vector<Triangle> make_torus(int nu, int nv, float R, float r, vec3 center, float tilt) { // procedural torus with exact vertex normals
    auto rotx = [tilt](vec3 p){ float c=std::cos(tilt), s=std::sin(tilt); return vec3{p.x, p.y*c-p.z*s, p.y*s+p.z*c}; };
    auto vert = [&](int i, int j, vec3 &P, vec3 &N) {
        float u = 2*pi*i/nu, v = 2*pi*j/nv;
        N = rotx({std::cos(v)*std::cos(u), std::sin(v), std::cos(v)*std::sin(u)});
        P = rotx({(R+r*std::cos(v))*std::cos(u), r*std::sin(v), (R+r*std::cos(v))*std::sin(u)}) + center;
    };
    std::vector<Triangle> tris;
    tris.reserve(size_t(nu)*nv*2);
    for (int i=0;i<nu;i++)
        for (int j=0;j<nv;j++) {
            vec3 p00,n00,p10,n10,p01,n01,p11,n11;
            vert(i,         j,         p00,n00); vert((i+1)%nu, j,         p10,n10);
            vert(i,         (j+1)%nv,  p01,n01); vert((i+1)%nu, (j+1)%nv,  p11,n11);
            tris.push_back({p00,p10,p11, n00,n10,n11});  // two triangles per quad of the (u,v) grid
            tris.push_back({p00,p11,p01, n00,n11,n01});
        }
    return tris;
}

int main(int argc, char **argv) {
    g_mesh.tris = argc>1 ? load_obj(argv[1])             // render an external model if given,
                         : make_torus(220, 90, 2.3f, 0.8f, {-3.6f, 1.7f, -10.f}, -1.0f); // otherwise a procedural torus
    g_mesh.build();
    std::fprintf(stderr, "mesh: %zu triangles, BVH: %zu nodes\n", g_mesh.tris.size(), g_mesh.nodes.size());

    constexpr int   width  = 1024;
    constexpr int   height = 768;
    constexpr int   spp    = 64;     // Monte-Carlo samples per pixel (also gives free anti-aliasing)
    constexpr float fov    = 1.05;   // 60 degrees field of view in radians
    std::vector<vec3> framebuffer(width*height);
#pragma omp parallel for
    for (int pix = 0; pix<width*height; pix++) { // actual rendering loop
        const int i = pix%width, j = pix/width;
        vec3 color{0,0,0};
        for (int s = 0; s<spp; s++) {
            RNG rng{ seed(pix, s) | 1 };
            float dir_x =  (i + rng()) -  width/2.f;       // jitter the sample inside the pixel
            float dir_y = -(j + rng()) + height/2.f;       // this flips the image at the same time
            float dir_z = -height/(2.f*std::tan(fov/2.f));
            color = color + path_trace(vec3{0,0,0}, vec3{dir_x, dir_y, dir_z}.normalized(), rng);
        }
        framebuffer[pix] = color*(1.f/spp);
    }

    std::ofstream ofs("./out.ppm", std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (vec3 &color : framebuffer)
        for (int chan : {0,1,2}) {
            float v = std::pow(std::min(1.f, std::max(0.f, color[chan])), 1.f/2.2f); // clamp + linear->sRGB gamma
            ofs << (char)(255*v);
        }
    return 0;
}
