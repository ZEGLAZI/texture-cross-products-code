// Interpolate each pixel among the adjacent bin centres, clamping outside the grid.
// No supervised colour parameters are estimated. Compile with -O3, without fast-math.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

struct Bin { int left; double right_weight; };

static Bin interval(double x, const std::array<double,4>& k) {
    if (x <= k[0]) return {0,0.};
    if (x >= k[3]) return {2,1.};
    int lo = x < k[1] ? 0 : x < k[2] ? 1 : 2;
    return {lo,(x-k[lo])/(k[lo+1]-k[lo])};
}
static void vote(double *h,Bin x,Bin y,Bin z) {
    const int base=16*x.left+4*y.left+z.left;
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        const double w=(i?x.right_weight:1.-x.right_weight)*(j?y.right_weight:1.-y.right_weight);
        h[base+16*i+4*j] += w*(1.-z.right_weight);
        h[base+16*i+4*j+1] += w*z.right_weight;
    }
}
static double lab_f(double t) {
    constexpr double delta=6./29.;
    return t>delta*delta*delta ? std::cbrt(t) : t/(3.*delta*delta)+4./29.;
}
extern "C" void palettes(const uint8_t *rgb, int n, double *out) {
    const std::array<double,4> rgb_centres{31.5,95.5,159.5,223.5};
    const std::array<double,4> light{12.5,37.5,62.5,87.5}, chroma{-45.,-10.,10.,45.};
    std::array<double,256> linear{}, midcdf{};
    std::array<Bin,256> rgb_bins{}, relative_bins{};
    std::array<int,256> freq{};
    for(int i=0;i<256;++i) {
        const double v=i/255.;
        linear[i]=v<=.04045 ? v/12.92 : std::pow((v+.055)/1.055,2.4);
        rgb_bins[i]=interval(i,rgb_centres);
    }
    struct RelativePixel { uint8_t luminance; Bin a,b; };
    std::vector<RelativePixel> relative(n);
    std::fill(out,out+5*64,0.);
    for(int i=0;i<n;++i) {
        const int r=rgb[3*i],g=rgb[3*i+1],b=rgb[3*i+2];
        ++out[16*(r/64)+4*(g/64)+b/64];
        vote(out+64,rgb_bins[r],rgb_bins[g],rgb_bins[b]);
        const double R=linear[r],G=linear[g],B=linear[b];
        const double fx=lab_f((.4124564*R+.3575761*G+.1804375*B)/.95047);
        const double fy=lab_f(.2126729*R+.7151522*G+.0721750*B);
        const double fz=lab_f((.0193339*R+.1191920*G+.9503041*B)/1.08883);
        const double l=116.*fy-16., a=500.*(fx-fy), bb=200.*(fy-fz);
        const Bin lb=interval(l,light), ab=interval(a,chroma), bbin=interval(bb,chroma);
        const auto near=[](Bin v){return v.left+int(v.right_weight>=.5);};
        ++out[128+16*near(lb)+4*near(ab)+near(bbin)];
        vote(out+192,lb,ab,bbin);
        const uint8_t level=static_cast<uint8_t>(std::clamp(int(std::lround(l*2.55)),0,255));
        ++freq[level]; relative[i]={level,ab,bbin};
    }
    int count=0;
    for(int i=0;i<256;++i) {
        midcdf[i]=100.*(count+.5*freq[i])/n;
        relative_bins[i]=interval(midcdf[i],light);
        count+=freq[i];
    }
    for(const auto &p:relative) vote(out+256,relative_bins[p.luminance],p.a,p.b);
    for(int i=0;i<320;++i) out[i]=std::sqrt(out[i]/n);
}
