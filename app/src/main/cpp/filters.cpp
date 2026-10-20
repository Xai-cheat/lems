// RetroLens native filters (C++ / NDK)
// Filter ids: 0 = none, 1 = CYBER_NEON, 2 = SEPIA, 3 = NEGATIVE,
//               4 = THERMAL, 5 = NOIR, 6 = MATRIX, 7 = SUNSET,
//               8 = COMIC, 9 = ICE, 10 = VAPOR, 11 = GOLD, 12 = TOXIC, 13 = NIGHT, 14 = FILM
#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <thread>
#include <vector>

static inline uint8_t clamp8(float v) { return v < 0 ? 0 : (v > 255 ? 255 : (uint8_t)v); }

static inline void applyOne(uint8_t* p, int f) {
    float r = p[0], g = p[1], b = p[2];
    float l = 0.299f * r + 0.587f * g + 0.114f * b;
    switch (f) {
        case 1:  // cyber neon: pink/purple glow
            p[0] = clamp8(0.6f * r + 0.5f * l + 40);
            p[1] = clamp8(0.35f * g);
            p[2] = clamp8(0.7f * b + 0.6f * l + 60);
            break;
        case 2:  // sepia
            p[0] = clamp8(0.393f * r + 0.769f * g + 0.189f * b);
            p[1] = clamp8(0.349f * r + 0.686f * g + 0.168f * b);
            p[2] = clamp8(0.272f * r + 0.534f * g + 0.131f * b);
            break;
        case 3:  // negative
            p[0] = 255 - p[0];
            p[1] = 255 - p[1];
            p[2] = 255 - p[2];
            break;
        case 4: {  // thermal camera: luminance -> heat map
            float t = l / 255.0f;
            float rr = 1.5f - std::fabs(4.0f * t - 3.0f);
            float gg = 1.5f - std::fabs(4.0f * t - 2.0f);
            float bb = 1.5f - std::fabs(4.0f * t - 1.0f);
            p[0] = clamp8(255.0f * std::min(1.0f, std::max(0.0f, rr)));
            p[1] = clamp8(255.0f * std::min(1.0f, std::max(0.0f, gg)));
            p[2] = clamp8(255.0f * std::min(1.0f, std::max(0.0f, bb)));
            break;
        }
        case 5: {  // noir: high-contrast black & white
            uint8_t v = clamp8((l - 128.0f) * 1.4f + 128.0f);
            p[0] = p[1] = p[2] = v;
            break;
        }
        case 6:  // matrix: green terminal
            p[0] = clamp8(0.10f * l);
            p[1] = clamp8(1.20f * l + 10);
            p[2] = clamp8(0.15f * l);
            break;
        case 7:  // sunset: warm orange/pink
            p[0] = clamp8(0.90f * r + 0.40f * l + 30);
            p[1] = clamp8(0.55f * g + 0.15f * l);
            p[2] = clamp8(0.50f * b + 0.20f * l + 20);
            break;
        case 8: {  // comic: posterized, extra-saturated colors
            float sat = 1.6f;
            float rr = l + (r - l) * sat, gg = l + (g - l) * sat, bb = l + (b - l) * sat;
            p[0] = clamp8(std::floor(rr / 64.0f) * 64.0f + 32.0f);
            p[1] = clamp8(std::floor(gg / 64.0f) * 64.0f + 32.0f);
            p[2] = clamp8(std::floor(bb / 64.0f) * 64.0f + 32.0f);
            break;
        }
        case 9:  // ice: cold blue-white
            p[0] = clamp8(0.45f * r + 0.25f * l);
            p[1] = clamp8(0.70f * g + 0.30f * l + 15);
            p[2] = clamp8(0.80f * b + 0.55f * l + 40);
            break;
        case 10: {  // vapor: teal shadows -> pink highlights
            float t = l / 255.0f;
            p[0] = clamp8(255.0f * (0.10f + 0.90f * t));
            p[1] = clamp8(255.0f * (0.85f - 0.60f * t));
            p[2] = clamp8(255.0f * (0.90f - 0.10f * t));
            break;
        }
        case 11:  // gold: warm metallic
            p[0] = clamp8(1.10f * l + 25);
            p[1] = clamp8(0.85f * l + 10);
            p[2] = clamp8(0.35f * l);
            break;
        case 12:  // toxic: acid yellow-green
            p[0] = clamp8(0.70f * l + 0.30f * r);
            p[1] = clamp8(1.10f * l + 20);
            p[2] = clamp8(0.15f * b);
            break;
        case 13: {  // night vision: brighten the dark, green phosphor
            float g2 = 255.0f * std::sqrt(l / 255.0f);
            p[0] = clamp8(0.18f * g2);
            p[1] = clamp8(g2 + 8);
            p[2] = clamp8(0.25f * g2);
            break;
        }
        case 14:  // film: faded blacks, warm tint, softer contrast
            p[0] = clamp8((0.90f * r + 0.10f * g) * 0.85f + 34);
            p[1] = clamp8((0.88f * g + 0.06f * b) * 0.85f + 26);
            p[2] = clamp8((0.72f * b + 0.10f * r) * 0.85f + 30);
            break;
        default: break;
    }
}

static bool insideQuad(const float* q, float x, float y) {
    int pos = 0, neg = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) & 3;
        float c = (q[j * 2] - q[i * 2]) * (y - q[i * 2 + 1]) -
                  (q[j * 2 + 1] - q[i * 2 + 1]) * (x - q[i * 2]);
        if (c > 0) pos++; else if (c < 0) neg++;
    }
    return pos == 0 || neg == 0;
}

// Layers, from bottom to top:
//   1. the whole screen        -> screenFilter
//   2. shockwave reveal        -> pixels OUTSIDE the circle (revealX, revealY, revealR)
//                                 still show prevFilter (the filter before the slingshot shot)
//   3. the portal quad         -> portalFilter (or the clean image if it equals the screen filter)
// Pass revealR < 0 (or prevFilter < 0) to disable the reveal.
extern "C" JNIEXPORT void JNICALL
Java_com_retrolens_app_MainActivity_applyFilters(JNIEnv* env, jobject, jobject bitmap,
                                                 jint screenFilter, jint portalFilter,
                                                 jfloatArray quadArr,
                                                 jint prevFilter, jfloat revealX, jfloat revealY,
                                                 jfloat revealR,
                                                 jint portalPrev, jfloat portalMix) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;

    float q[8];
    bool hasQuad = quadArr != nullptr && env->GetArrayLength(quadArr) >= 8;
    if (hasQuad) env->GetFloatArrayRegion(quadArr, 0, 8, q);

    const bool revealOn = revealR >= 0.0f && prevFilter >= 0;
    const int outsideFilter = revealOn ? prevFilter : screenFilter;
    // Inside the portal: show the portal filter, or the clean image if it
    // matches the screen filter (this gives the "portal" inversion effect).
    const int inside = (portalFilter == screenFilter) ? 0 : portalFilter;
    // right after the portal filter changes, fade from the old one to the new one
    const bool mixOn = hasQuad && portalPrev >= 0 && portalMix < 0.999f;
    const int prevInside = (portalPrev == screenFilter) ? 0 : portalPrev;
    const float mixT = portalMix < 0.0f ? 0.0f : (portalMix > 1.0f ? 1.0f : portalMix);

    if (!hasQuad && screenFilter == 0 && outsideFilter == 0) return;  // nothing to draw

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return;

    const int W = (int)info.width, H = (int)info.height;
    int minX = 0, maxX = W - 1, minY = 0, maxY = H - 1;
    if (hasQuad) {
        float x0 = q[0], x1 = q[0], y0 = q[1], y1 = q[1];
        for (int i = 1; i < 4; i++) {
            x0 = std::min(x0, q[i * 2]); x1 = std::max(x1, q[i * 2]);
            y0 = std::min(y0, q[i * 2 + 1]); y1 = std::max(y1, q[i * 2 + 1]);
        }
        minX = std::max(0, (int)x0); maxX = std::min(W - 1, (int)x1);
        minY = std::max(0, (int)y0); maxY = std::min(H - 1, (int)y1);
    }

    int ys = 0, ye = H, xs = 0, xe = W;
    if (!revealOn && screenFilter == 0 && hasQuad) {  // only the portal needs work
        if (maxX < minX || maxY < minY) { AndroidBitmap_unlockPixels(env, bitmap); return; }
        ys = minY; ye = maxY + 1; xs = minX; xe = maxX + 1;
    }

    const float r2 = revealR * revealR;
    const size_t stride = info.stride;
    uint8_t* base = (uint8_t*)pixels;
    // rows are independent, so split them across a few threads
    auto work = [&](int ya, int yb) {
        for (int y = ya; y < yb; y++) {
            uint8_t* row = base + (size_t)y * stride;
            const float dy = (float)y - revealY;
            const float dy2 = dy * dy;
            const bool rowMayBeInside = hasQuad && y >= minY && y <= maxY;
            for (int x = xs; x < xe; x++) {
                int f = screenFilter;
                if (revealOn) {
                    const float dx = (float)x - revealX;
                    if (dx * dx + dy2 > r2) f = prevFilter;
                }
                if (rowMayBeInside && x >= minX && x <= maxX && insideQuad(q, (float)x, (float)y)) {
                    if (mixOn) {
                        uint8_t* px = row + (size_t)x * 4;
                        uint8_t b[4] = {px[0], px[1], px[2], px[3]};
                        if (prevInside) applyOne(b, prevInside);
                        if (inside) applyOne(px, inside);
                        for (int ch = 0; ch < 3; ch++)
                            px[ch] = (uint8_t)((float)b[ch] + ((float)px[ch] - (float)b[ch]) * mixT);
                        continue;
                    }
                    f = inside;
                }
                if (f) applyOne(row + (size_t)x * 4, f);
            }
        }
    };
    const int rows = ye - ys;
    if (rows < 160) {
        work(ys, ye);
    } else {
        const int nT = 3;
        std::vector<std::thread> th;
        for (int t = 1; t < nT; t++) {
            int a0 = ys + rows * t / nT, a1 = ys + rows * (t + 1) / nT;
            th.emplace_back(work, a0, a1);
        }
        work(ys, ys + rows / nT);
        for (auto& t : th) t.join();
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}
