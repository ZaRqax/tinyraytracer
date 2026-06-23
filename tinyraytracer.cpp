#include <tuple>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdint>
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

int main() {
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
