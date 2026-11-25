// Local CIELAB colour pairs and their exact expectation after permutation without replacement.
#include "palettes.cpp"
static int pair_index(int a,int b) {
    const int lo=std::min(a,b),hi=std::max(a,b);
    return hi*(hi+1)/2+lo;
}
extern "C" void spatial(const uint8_t* rgb,int height,int width,double* out) {
    const int n=height*width;
    std::array<double,256> linear{};
    for(int i=0;i<256;++i) {
        const double v=i/255.;linear[i]=v<=.04045?v/12.92:std::pow((v+.055)/1.055,2.4);
    }
    std::array<int,27> population{};
    std::vector<uint8_t> labels(n);
    std::fill(out,out+756,0.);
    for(int i=0;i<n;++i) {
        const double r=linear[rgb[3*i]],g=linear[rgb[3*i+1]],b=linear[rgb[3*i+2]];
        const double fx=lab_f((.4124564*r+.3575761*g+.1804375*b)/.95047);
        const double fy=lab_f(.2126729*r+.7151522*g+.0721750*b);
        const double fz=lab_f((.0193339*r+.1191920*g+.9503041*b)/1.08883);
        const double l=116.*fy-16.,a=500.*(fx-fy),bb=200.*(fy-fz);
        const int li=(l>=100./3.)+(l>=200./3.);
        const auto c=[](double v){return (v>=-12.)+(v>=12.);};
        const int index=9*li+3*c(a)+c(bb);labels[i]=index;++population[index];
    }
    for(int distance:{2,8}) {
        const std::array<std::array<int,2>,4> directions{{{0,distance},{distance,0},
            {distance,distance},{distance,-distance}}};
        for(auto direction:directions) {
            const int dy=direction[0],dx=direction[1];
            const int x0=std::max(0,-dx),x1=width-std::max(dx,0),y1=height-dy;
            std::array<int,378> counts{};
            for(int y=0;y<y1;++y) for(int x=x0;x<x1;++x)
                ++counts[pair_index(labels[y*width+x],labels[(y+dy)*width+x+dx])];
            const double norm=8.*(x1-x0)*y1;
            for(int j=0;j<378;++j) out[j]+=counts[j]/norm;
        }
    }
    const double possible=static_cast<double>(n)*(n-1);
    for(int j=0;j<27;++j) for(int i=0;i<=j;++i) {
        const double pairs=i==j?double(population[i])*(population[i]-1):
                     2.*population[i]*population[j];
        out[378+pair_index(i,j)]=std::sqrt(pairs/possible);
    }
    for(int j=0;j<378;++j) out[j]=std::sqrt(out[j]);
}
