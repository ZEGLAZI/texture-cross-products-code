// Signed multiscale achromatic contrast, with conditional colour measurements.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
using Triple=std::array<double,3>;
using Psi=std::array<double,10>;
constexpr double TOL=1.e-8;

struct Integral {
    int w, h;
    std::vector<Triple> prefix;
    Integral(int width,int height):w(width),h(height),prefix((w+1)*(h+1),Triple{}) {}
    void append(int y,int x,const Triple &pixel,Triple &row) {
        for(int k=0;k<3;++k) {
            row[k]+=pixel[k];
            prefix[(y+1)*(w+1)+x+1][k]=row[k]+prefix[y*(w+1)+x+1][k];
        }
    }
    double sum(int y0,int y1,int x0,int x1,int k) const {
        const int s=w+1;
        return (prefix[y1*s+x1][k]-prefix[y1*s+x0][k])-
               (prefix[y0*s+x1][k]-prefix[y0*s+x0][k]);
    }
    double haar(int y,int x,int r,int axis,int k) const {
        double left,right;
        if(axis==0) {
            left=sum(y-r,y+r,x-r,x,k);right=sum(y-r,y+r,x,x+r,k);
        } else {
            left=sum(y-r,y,x-r,x+r,k);right=sum(y,y+r,x-r,x+r,k);
        }
        return (right-left)/(2.*r*r);
    }
};
struct Stream {
    uint64_t state;
    uint64_t next() {
        uint64_t z=(state+=UINT64_C(0x9e3779b97f4a7c15));
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        return z^(z>>31);
    }
    void shuffle(std::vector<uint8_t>& flags) {
        for(size_t i=flags.size();i>1;--i) {
            const uint64_t bound=i, threshold=(uint64_t(0)-bound)%bound;
            uint64_t v;do { v=next(); } while(v<threshold);
            std::swap(flags[i-1], flags[v%bound]);
        }
    }
};
static double median(std::vector<double> v) {
    if(v.empty())return 0.;
    const size_t m=v.size()/2;
    auto it=v.begin()+m;
    std::nth_element(v.begin(),it,v.end());
    return (v.size()%2)?*it:.5*(*it+*std::max_element(v.begin(),it));
}
static void embed(double a,double b,double* p) {
    const double denom=1.+std::hypot(a,b);
    a/=denom;b/=denom;
    p[0]=a;p[1]=b;p[2]=a*a;p[3]=a*b;p[4]=b*b;
}
struct Site { int x,y; double fine,coarse; };
struct Tagged { Psi v; uint8_t flag; };
static void compute(const Integral& a,uint64_t seed,double*out,double*diagnostics) {
    // Output: 2 candidates x 3 statistics (aligned, unconditioned, permuted) x 30 coordinates.
    std::fill(out,out+180,0.);std::fill(diagnostics,diagnostics+36,0.);
    Stream rng{seed};int cell=0;
    const double mean1=a.sum(0,a.h,0,a.w,1)/(a.h*a.w);
    const double mean2=a.sum(0,a.h,0,a.w,2)/(a.h*a.w);
    for(int r:{2,4,8})for(int axis:{0,1}) {
        std::vector<Site> all;
        std::vector<double> fine,coarse;
        for(int y=2*r;y<=a.h-2*r;y+=2)for(int x=2*r;x<=a.w-2*r;x+=2) {
            const double f=a.haar(y,x,r,axis,0),c=a.haar(y,x,2*r,axis,0);
            all.push_back({x,y,f,c});fine.push_back(std::abs(f));
        }
        const double cut=median(fine);fine.clear();
        std::vector<Site> kept;
        for(auto site:all) {
            if(std::abs(site.fine)<cut || std::abs(site.fine)<=TOL || std::abs(site.coarse)<=TOL)continue;
            fine.push_back(std::abs(site.fine));coarse.push_back(std::abs(site.coarse));kept.push_back(site);
        }
        const double fmid=median(fine),cmid=median(coarse);
        std::array<std::vector<Tagged>,4> strata;
        for(auto site:kept) {
            const int x=site.x,y=site.y;
            const double sign=site.fine>0?1.:-1.;
            const auto key=2*(std::abs(site.fine)>fmid)+(std::abs(site.coarse)>cmid);
            Tagged t{};t.flag=(site.fine>0)==(site.coarse>0);
            embed(a.sum(y-r,y+r,x-r,x+r,1)/(4.*r*r)-mean1,
                  a.sum(y-r,y+r,x-r,x+r,2)/(4.*r*r)-mean2,t.v.data());
            embed(sign*a.haar(y,x,r,axis,1),sign*a.haar(y,x,r,axis,2),t.v.data()+5);
            strata[key].push_back(t);
        }
        const double N=kept.size();diagnostics[6*cell]=N;
        for(int g=0;g<4;++g) {
            const auto& sites=strata[g];diagnostics[6*cell+2+g]=sites.size();
            if(sites.empty())continue;
            std::vector<uint8_t> flags;int positive=0;
            for(const auto&s:sites){flags.push_back(s.flag);positive+=s.flag;}
            const double prob=double(positive)/sites.size();
            diagnostics[6*cell+1]+=positive/N;
            std::vector<double> shuffled(sites.size(),0.);
            for(int draw=0;draw<4;++draw) {
                rng.shuffle(flags);
                for(size_t i=0;i<flags.size();++i)shuffled[i]+=.25*flags[i];
            }
            for(size_t i=0;i<sites.size();++i){
                const auto& s=sites[i];
                for(int j=0;j<10;++j) {
                    const int index=(j/5)*90+5*cell+j%5;
                    const double val=s.v[j]/N;
                    out[index]+=4.*val*(s.flag-prob);
                    out[index+30]+=val;
                    out[index+60]+=4.*val*(shuffled[i]-prob);
                }
            }
        }
        ++cell;
    }
}
static Triple opponent(double r,double g,double b) {
    return {(r+g+b)/3.,(r-g)/std::sqrt(2.),(r+g-2.*b)/std::sqrt(6.)};
}
static double f_lab(double t) {
    constexpr double delta=6./29.;
    return t>delta*delta*delta?std::cbrt(t):t/(3.*delta*delta)+4./29.;
}
extern "C" void colour_describe(const uint8_t* rgb,int h,int w,uint64_t seed,double* lab,double* edges,double* diagnostics) {
    Integral a(w,h); std::array<double,256> lin{},logs{};
    for(int k=0;k<256;++k) {
        const double v=k/255.;lin[k]=v<=.04045?v/12.92:std::pow((v+.055)/1.055,2.4);
    }
    for(int k=0;k<256;++k)logs[k]=std::log(std::max(lin[k],lin[1]));
    Triple avg{},m2{},m3{};int n=0;
    for(int y=0;y<h;++y) {
        Triple row{};
        for(int x=0;x<w;++x) {
            const int i=(y*w+x)*3,ri=rgb[i],gi=rgb[i+1],bi=rgb[i+2];
            a.append(y,x,opponent(logs[ri],logs[gi],logs[bi]),row);
            const double r=lin[ri],g=lin[gi],b=lin[bi];
            const double fx=f_lab((.4124564*r+.3575761*g+.1804375*b)/.95047);
            const double fy=f_lab(.2126729*r+.7151522*g+.0721750*b);
            const double fz=f_lab((.0193339*r+.1191920*g+.9503041*b)/1.08883);
            const Triple val{116.*fy-16.,500.*(fx-fy),200.*(fy-fz)};++n;
            for(int k=0;k<3;++k){
                const double d=val[k]-avg[k],dn=d/n,increment=d*dn*(n-1);
                m3[k]+=increment*dn*(n-2)-3.*dn*m2[k];m2[k]+=increment;avg[k]+=dn;
            }
        }
    }
    for(int k=0;k<3;++k) {
        lab[3*k]=avg[k];lab[3*k+1]=std::sqrt(std::max(0.,m2[k]/n));lab[3*k+2]=std::cbrt(m3[k]/n);
    }
    compute(a,seed,edges,diagnostics);
}
// Testing entry point: linear-positive sensor values supplied as log RGB, before quantization.
extern "C" void from_log_rgb(const double* logs,int h,int w,uint64_t seed,double* edges,double* diagnostics){
    Integral a(w,h);
    for(int y=0;y<h;++y){Triple row{};for(int x=0;x<w;++x){
        const double*p=logs+3*(y*w+x);a.append(y,x,opponent(p[0],p[1],p[2]),row);
    }}
    compute(a,seed,edges,diagnostics);
}
