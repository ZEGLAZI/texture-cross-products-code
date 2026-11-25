// Independent implementation of CLBP S/M/C and of OCLBP (RGB-RGB).
// Coordinates follow the source papers; only centres with a complete neighbourhood enter.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int points = 24;
constexpr int radius = 3;
struct Offset { int x, y; double weight; };

std::array<Offset, 4> interpolation(int p) {
    const double angle = 2 * std::acos(-1.0) * p / points;
    double x = radius * std::cos(angle), y = -radius * std::sin(angle);
    if (std::abs(x - std::round(x)) < 1e-12) x = std::round(x);
    if (std::abs(y - std::round(y)) < 1e-12) y = std::round(y);
    int ix = std::floor(x), iy = std::floor(y);
    double fx = x - ix, fy = y - iy;
    return {{{ix, iy, (1-fx)*(1-fy)}, {ix+1, iy, fx*(1-fy)},
             {ix, iy+1, (1-fx)*fy}, {ix+1, iy+1, fx*fy}}};
}

struct Bits {
    int first = 0, previous = 0, ones = 0, changes = 0;
    void add(int bit, int p) {
        if (p == 0) first = bit;
        else changes += (bit != previous);
        ones += bit;
        previous = bit;
    }
    int riu2() const { return changes + (first != previous) <= 2 ? ones : points + 1; }
};

std::array<int, 256> rotation_map() {
    std::array<int, 256> output{}, labels{};
    labels.fill(-1);
    int count = 0;
    for (int i = 0; i < 256; i++) {
        int smallest = i, word = i;
        for (int b = 1; b < 8; b++) {
            word = (word >> 1) | ((word & 1) << 7);
            smallest = std::min(smallest, word);
        }
        if (labels[smallest] < 0) labels[smallest] = count++;
        output[i] = labels[smallest];
    }
    return output;
}
}

extern "C" void clbp(const uint8_t* gray, int h, int w, int32_t* hist) {
    const int n = (w-2*radius)*(h-2*radius);
    std::vector<double> delta(static_cast<size_t>(n)*points, 0.0);
    double total = 0, gray_sum = 0;
    for (int i = 0; i < h*w; i++) gray_sum += gray[i];
    for (int p = 0; p < points; p++) {
        double* plane = delta.data() + static_cast<size_t>(p)*n;
        const auto stencil = interpolation(p);
        for (const auto& v : stencil) {
            if (v.weight == 0) continue;
            int k = 0;
            for (int y = radius; y < h-radius; y++) {
                const uint8_t* center = gray + y*w + radius;
                const uint8_t* other = center + v.y*w + v.x;
                for (int x = 0; x < w-2*radius; x++, k++)
                    plane[k] += v.weight*(static_cast<int>(other[x])-center[x]);
            }
        }
        for (int k = 0; k < n; k++) total += std::abs(plane[k]);
    }
    const double magnitude_mean = total/(n*points), center_mean = gray_sum/(h*w);
    std::fill(hist, hist + 2*(points+2)*(points+2), 0);
    for (int y = radius, k=0; y < h-radius; y++)
        for (int x = radius; x < w-radius; x++, k++) {
            Bits sign, mag;
            for (int p = 0; p < points; p++) {
                double d = delta[static_cast<size_t>(p)*n+k];
                sign.add(d >= -1e-10, p);
                mag.add(std::abs(d) >= magnitude_mean-1e-12, p);
            }
            int c = gray[y*w+x] >= center_mean;
            hist[(sign.riu2()*(points+2)+mag.riu2())*2+c]++;
        }
}

extern "C" void oclbp(const uint8_t* rgb, int h, int w, int32_t* hist) {
    static const auto mapping = rotation_map();
    std::fill(hist, hist+432, 0);
    // Pair ordering: neighbour channel then centre channel, as in Fig. 2.
    const int samples[12][3] = {{1,0,0},{1,1,1},{1,2,2},{2,0,0},{2,1,1},{2,2,2},
                                {3,0,0},{3,1,1},{3,2,2},{1,0,1},{1,0,2},{1,1,2}};
    for (int b = 0; b < 12; b++) {
        int r = samples[b][0], nc = samples[b][1], cc = samples[b][2];
        int d = r == 3 ? 2 : 1;
        const int dx[] = {r,d,0,-d,-r,-d,0,d}, dy[] = {0,-d,-r,-d,0,d,r,d};
        int offset[8];
        for (int j=0; j<8; j++) offset[j]=(dy[j]*w+dx[j])*3+nc;
        for (int y = r; y < h-r; y++)
            for (int x = r; x < w-r; x++) {
                const uint8_t* pixel = rgb + (y*w+x)*3;
                int word=0;
                for (int j=0; j<8; j++) word |= (pixel[offset[j]] >= pixel[cc]) << j;
                hist[b*36+mapping[word]]++;
            }
    }
}

// Pairwise distances, computed once. Training-fold masks are applied in Python.
// A fixed global scaling of the chi-square distance cancels in nearest-neighbour decisions.
extern "C" void distances(const double* x, int n, int k, int chi, int threads, double* out) {
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 3)
    for (int i=0; i<n; i++) {
        const double* a=x+static_cast<size_t>(i)*k;
        out[static_cast<size_t>(i)*n+i]=0.0;
        for (int j=0; j<i; j++) {
            const double* b=x+static_cast<size_t>(j)*k;
            double sum=0;
            if (chi) {
                #pragma omp simd reduction(+:sum)
                for (int l=0; l<k; l++) {
                    double d=a[l]-b[l];
                    sum+=d*d/(a[l]+b[l]+1e-30);
                }
            } else {
                #pragma omp simd reduction(+:sum)
                for (int l=0; l<k; l++) sum+=std::abs(a[l]-b[l]);
            }
            out[static_cast<size_t>(i)*n+j]=out[static_cast<size_t>(j)*n+i]=sum;
        }
    }
}
