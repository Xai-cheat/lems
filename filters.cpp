// RetroLens native filters (C++ / NDK)
// Filter ids: 0 = none, 1 = CYBER_NEON, 2 = SEPIA, 3 = NEGATIVE
#include <jni.h>
#include <android/bitmap.h>
#include <algorithm>
#include <cstdint>

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

extern "C" JNIEXPORT void JNICALL
Java_com_retrolens_app_MainActivity_applyFilters(JNIEnv* env, jobject, jobject bitmap,
                                                 jint screenFilter, jint portalFilter,
                                                 jfloatArray quadArr) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) return;

    float q[8];
    bool hasQuad = quadArr != nullptr;
    if (hasQuad) env->GetFloatArrayRegion(quadArr, 0, 8, q);

    int minX = 0, maxX = (int)info.width - 1, minY = 0, maxY = (int)info.height - 1;
    if (hasQuad) {
        float x0 = q[0], x1 = q[0], y0 = q[1], y1 = q[1];
        for (int i = 1; i < 4; i++) {
            x0 = std::min(x0, q[i * 2]); x1 = std::max(x1, q[i * 2]);
            y0 = std::min(y0, q[i * 2 + 1]); y1 = std::max(y1, q[i * 2 + 1]);
        }
        minX = std::max(0, (int)x0); maxX = std::min((int)info.width - 1, (int)x1);
        minY = std::max(0, (int)y0); maxY = std::min((int)info.height - 1, (int)y1);
    }
    // Inside the portal: show the portal filter, or the clean image if it
    // matches the screen filter (this gives the "portal" inversion effect).
    int inside = (portalFilter == screenFilter) ? 0 : portalFilter;

    for (uint32_t y = 0; y < info.height; y++) {
        uint8_t* row = (uint8_t*)pixels + y * info.stride;
        for (uint32_t x = 0; x < info.width; x++) {
            int f = screenFilter;
            if (hasQuad && (int)x >= minX && (int)x <= maxX && (int)y >= minY && (int)y <= maxY &&
                insideQuad(q, (float)x, (float)y))
                f = inside;
            if (f) applyOne(row + x * 4, f);
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}
