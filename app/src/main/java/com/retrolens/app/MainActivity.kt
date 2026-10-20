package com.retrolens.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.hypot

class MainActivity : ComponentActivity() {

    external fun applyFilters(bmp: Bitmap, screenFilter: Int, portalFilter: Int, quad: FloatArray?)

    companion object {
        init { System.loadLibrary("retrolens") }
        val NAMES = arrayOf("NONE", "CYBER_NEON", "SEPIA", "NEGATIVE", "THERMAL", "NOIR", "MATRIX", "SUNSET")
    }

    private lateinit var imageView: ImageView
    @Volatile private var landmarker: HandLandmarker? = null
    @Volatile private var usingGpu = false
    @Volatile private var switching = false
    private var errView: TextView? = null

    private val exec = Executors.newSingleThreadExecutor()   // camera frames -> drawing

    // reusable output buffers (no per-frame allocation = no GC stutter)
    private val ring = arrayOfNulls<Bitmap>(3)
    private var ringIdx = 0

    // frame size, written by the drawing thread, read by the result callback
    @Volatile private var frameW = 0f
    @Volatile private var frameH = 0f

    // gesture state: written by MediaPipe's result callback, read by the drawing thread
    @Volatile private var screenFilter = 0
    @Volatile private var portalFilter = 1
    @Volatile private var quad: FloatArray? = null
    @Volatile private var line: Pair<PointF, PointF>? = null
    @Volatile private var handsPx: List<List<PointF>> = emptyList()
    @Volatile private var msg = ""
    @Volatile private var msgUntil = 0L
    @Volatile private var charge = 0f
    @Volatile private var flashUntil = 0L

    // code log shown at the bottom of the screen: (language, code line)
    private val logLines = ArrayList<Pair<String, String>>()
    private fun logEv(lang: String, code: String) {
        synchronized(logLines) {
            logLines.add(Pair(lang, code))
            while (logLines.size > 4) logLines.removeAt(0)
        }
    }

    // result-callback only
    private var lastSwitch = 0L
    private var wasSling = false
    private var lastSlingAt = 0L
    private var lastSlingLine: Pair<PointF, PointF>? = null
    private var lastCharge = 0f
    private var prevQuad: FloatArray? = null
    private var hadQuad = false
    private var lastHandCount = -1
    private var lastHandLog = 0L

    // drawing-thread only
    private var lastTs = 0L
    private var lastDetSent = 0L
    private var dispQuad: FloatArray? = null
    private var dispLine: FloatArray? = null
    private var dispDots: FloatArray? = null

    private val green = Paint().apply { color = Color.rgb(60, 255, 60); strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true }
    private val dot = Paint().apply { color = Color.rgb(60, 2  val nowMs = System.currentTimeMillis()
            if (lm != null && !switching && nowMs - lastDetSent >= 40) {
                lastDetSent = nowMs
                val small = Bitmap.createScaledBitmap(bmp, bmp.width / 2, bmp.height / 2, true)
                lm.detectAsync(BitmapImageBuilder(small).build(), nextTs())
            }

            // glide the overlay toward the newest detection so it moves smoothly
            val tq = quad
            val dq = dispQuad
            val q = if (tq != null && dq != null) FloatArray(8) { dq[it] + (tq[it] - dq[it]) * 0.45f } else tq
            dispQuad = q
            val tl = line?.let { floatArrayOf(it.first.x, it.first.y, it.second.x, it.second.y) }
            val dl = dispLine
            val ln = if (tl != null && dl != null) FloatArray(4) { dl[it] + (tl[it] - dl[it]) * 0.45f } else tl
            dispLine = ln
            val hs = handsPx
            val ch = charge

            applyFilters(bmp, screenFilter, portalFilter, q)

            // white flash when the slingshot fires
            if (nowMs < flashUntil) c.drawRect(0f, 0f, w, h, flashPaint)

            q?.let {
                val p = Path().apply { moveTo(it[0], it[1]); lineTo(it[2], it[3]); lineTo(it[4], it[5]); lineTo(it[6], it[7]); close() }
                c.drawPath(p, green)
                val lx = it[2].coerceIn(4f, w * 0.55f)
                drawFit(c, "PORTAL: ${NAMES[portalFilter]}", lx, minOf(it[3], it[5]) - 20, text, w - lx - 4f)
            }
            ln?.let { c.drawLine(it[0], it[1], it[2], it[3], green) }

            // slingshot charge bar (like the PC version): "78% -> NEGATIVE"
            if (ln != null && ch > 0f) {
                val mx = (ln[0] + ln[2]) / 2f
                val my = (ln[1] + ln[3]) / 2f
                val bw = w * 0.34f
                val bh = h * 0.018f
                val bx = (mx - bw / 2f).coerceIn(4f, w - bw - 4f)
                val by = (my - bh - 36f).coerceIn(70f, h - 240f)
                c.drawRect(bx, by, bx + bw, by + bh, barBg)
                c.drawRect(bx, by, bx + bw * ch, by + bh, dot)
                drawFit(c, "${(ch * 100).toInt()}% -> ${NAMES[portalFilter]}", bx, by - 8f, text, w - bx - 4f)
            }

            // finger dots, smoothed
            val td = FloatArray(hs.size * 4)
            hs.forEachIndexed { k, hd -> td[k * 4] = hd[4].x; td[k * 4 + 1] = hd[4].y; td[k * 4 + 2] = hd[8].x; td[k * 4 + 3] = hd[8].y }
            val pd = dispDots
            val dd = if (pd != null && pd.size == td.size) FloatArray(td.size) { pd[it] + (td[it] - pd[it]) * 0.5f } else td
            dispDots = dd
            for (k in 0 until dd.size / 4) {
                c.drawCircle(dd[k * 4], dd[k * 4 + 1], 10f, dot)
                c.drawCircle(dd[k * 4 + 2], dd[k * 4 + 3], 10f, dot)
            }

            // code log panel at the bottom: what the app is doing right now
            val base = w * 0.037f
            val lh = base * 1.5f
            val panelTop = h - (lh * 4f + 10f)
            c.drawRect(0f, panelTop, w, h, panelBg)
            val snap = synchronized(logLines) { ArrayList(logLines) }
            snap.forEachIndexed { k, entry ->
                val prefix = if (entry.first == "C#") "C#  " else "C++ "
                // measure at the base size first, then shrink to fit
                mono.textSize = base
                val full = mono.measureText(prefix + entry.second)
                val maxW = w - 24f
                mono.textSize = if (full > maxW) base * maxW / full else base
                val y = panelTop + 6f + lh * (k + 1) - lh * 0.25f
                val pw = mono.measureText(prefix)
                mono.color = if (entry.first == "C#") Color.rgb(197, 134, 192) else Color.rgb(86, 156, 214)
                c.drawText(prefix, 12f, y, mono)
                mono.color = Color.rgb(120, 255, 120)
                c.drawText(entry.second, 12f + pw, y, mono)
            }

            if (System.currentTimeMillis() < msgUntil) drawFit(c, msg, 20f, panelTop - 16f, text, w - 40f)

            runOnUiThread { imageView.setImageBitmap(bmp) }
        } catch (e: Throwable) {
            runCatching { proxy.close() }
            Log.e("RetroLens", "frame error", e)
            if (!switching) runOnUiThread { errView?.text = "ERR: $e" }
        }
    }

    // ---------- MediaPipe result callback: decides gestures ----------
    private fun smoothQuad(n: FloatArray?): FloatArray? {
        val p = prevQuad
        val out = if (n != null && p != null) FloatArray(8) { 0.4f * p[it] + 0.6f * n[it] } else n
        prevQuad = out
        return out
    }

    private fun onResult(result: HandLandmarkerResult) {
        try {
            val w = frameW; val h = frameH
            if (w == 0f || h == 0f) return

            // hands as pixel-space points, sorted left-to-right on screen
            val hands = result.landmarks()
                .map { lm -> lm.map { PointF(it.x() * w, it.y() * h) } }
                .sortedBy { it[0].x }

            val now = System.currentTimeMillis()

            if (hands.size != lastHandCount && now - lastHandLog > 700) {
                lastHandCount = hands.size
                lastHandLog = now
                logEv("C#", "hands.Count = ${hands.size};")
            }

            var newQuad: FloatArray? = null
            var newLine: Pair<PointF, PointF>? = null
            var sling = false
            var newCharge = 0f

            if (hands.size == 2) {
                val a = hands[0]; val b = hands[1]

                // slingshot: one hand makes a peace sign, the other pinches - in EITHER order
                var peace: List<PointF>? = null
                var pinch: List<PointF>? = null
                if (isPeace(a) && isPinch(b)) { peace = a; pinch = b }
                else if (isPeace(b) && isPinch(a)) { peace = b; pinch = a }

                if (peace != null && pinch != null) {
                    sling = true
                    val from = mid(peace[8], peace[12])   // between the two peace fingers
                    val to = mid(pinch[4], pinch[8])      // the pinch point
                    newLine = Pair(from, to)
                    newCharge = (d(from, to) / (w * 0.6f)).coerceIn(0f, 1f)
                    lastSlingAt = now
                    lastSlingLine = newLine
                    lastCharge = newCharge
                    note("KETAPEL - LEPAS UNTUK TEMBAK!", 400)
                } else if (isPinch(a) && isPinch(b)) {
                    newLine = Pair(mid(a[4], a[8]), mid(b[4], b[8]))
                    if (now - lastSwitch > 1000) {
                        lastSwitch = now
                        portalFilter = portalFilter % (NAMES.size - 1) + 1
                        note("PORTAL: ${NAMES[portalFilter]}", 1000)
                        logEv("C#", "Portal.Filter = ${NAMES[portalFilter]};")
                    }
                } else {
                    // portal = frame made by both hands (thumb + index tips)
                    newQuad = floatArrayOf(a[4].x, a[4].y, a[8].x, a[8].y, b[8].x, b[8].y, b[4].x, b[4].y)
                }
            } else if (hands.size == 1 && isOpenPalm(hands[0]) && screenFilter != 0) {
                screenFilter = 0
                note("RESET - filter layar dimatikan", 1200)
                logEv("C#", "Screen.Filter = NONE; // palm reset")
            }

            // tolerate brief detection drop-outs so the line doesn't flicker / misfire
            if (!sling && wasSling && now - lastSlingAt < 300) {
                sling = true
                newLine = lastSlingLine
                newCharge = lastCharge
            }

            if (sling && !wasSling) logEv("C#", "if (L.Peace && R.Pinch) Sling.Aim();")

            // slingshot released -> whole screen takes the portal filter
            if (wasSling && !sling) {
                screenFilter = portalFilter
                flashUntil = now + 180
                note("KETAPEL DITEMBAK! Layar: ${NAMES[screenFilter]}", 1500)
                logEv("C#", "Screen.Filter = ${NAMES[screenFilter]}; // BANG")
                logEv("C++", "applyFilters(bmp, ${NAMES[screenFilter]});")
            }
            wasSling = sling

            if (newQuad != null && !hadQuad) logEv("C++", "drawQuad(thumbL,idxL,idxR,thumbR);")
            if (newQuad == null && hadQuad) logEv("C#", "Portal.Close();")
            hadQuad = newQuad != null

            charge = if (sling) newCharge else 0f
            quad = smoothQuad(newQuad)
            line = newLine
            handsPx = hands
        } catch (e: Throwable) {
            Log.e("RetroLens", "result error", e)
        }
    }

    private fun note(s: String, ms: Long) { msg = s; msgUntil = System.currentTimeMillis() + ms }

    // ---- gesture helpers (landmark indices from MediaPipe Hands) ----
    private fun d(a: PointF, b: PointF) = hypot(a.x - b.x, a.y - b.y)
    private fun mid(a: PointF, b: PointF) = PointF((a.x + b.x) / 2f, (a.y + b.y) / 2f)
    private fun size(h: List<PointF>) = d(h[0], h[9])
    private fun isPinch(h: List<PointF>) = d(h[4], h[8]) / size(h) < 0.4f
    private fun up(h: List<PointF>, tip: Int, pip: Int, k: Float = 1.15f) = d(h[0], h[tip]) > d(h[0], h[pip]) * k
    private fun isPeace(h: List<PointF>) = up(h, 8, 6, 1.1f) && up(h, 12, 10, 1.1f) && !up(h, 16, 14, 1.25f) && !up(h, 20, 18, 1.25f)
    private fun isOpenPalm(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && up(h, 16, 14) && up(h, 20, 18)
}
 4 + 3] = hd[8].y }
            val pd = dispDots
            val dd = if (pd != null && pd.size == td.size) FloatArray(td.size) { pd[it] + (td[it] - pd[it]) * 0.5f } else td
            dispDots = dd
            for (k in 0 until dd.size / 4) {
                c.drawCircle(dd[k * 4], dd[k * 4 + 1], 10f, dot)
                c.drawCircle(dd[k * 4 + 2], dd[k * 4 + 3], 10f, dot)
            }

            if (System.currentTimeMillis() < msgUntil) c.drawText(msg, 20f, h - 120f, text)

            runOnUiThread { imageView.setImageBitmap(bmp) }
        } catch (e: Throwable) {
            runCatching { proxy.close() }
            Log.e("RetroLens", "frame error", e)
            if (!switching) runOnUiThread { errView?.text = "ERR: $e" }
        }
    }

    // ---------- MediaPipe result callback: decides gestures ----------
    private fun smoothQuad(n: FloatArray?): FloatArray? {
        val p = prevQuad
        val out = if (n != null && p != null) FloatArray(8) { 0.4f * p[it] + 0.6f * n[it] } else n
        prevQuad = out
        return out
    }

    private fun onResult(result: HandLandmarkerResult) {
        try {
            val w = frameW; val h = frameH
            if (w == 0f || h == 0f) return

            // hands as pixel-space points, sorted left-to-right on screen
            val hands = result.landmarks()
                .map { lm -> lm.map { PointF(it.x() * w, it.y() * h) } }
                .sortedBy { it[0].x }

            val now = System.currentTimeMillis()
            var newQuad: FloatArray? = null
            var newLine: Pair<PointF, PointF>? = null
            var sling = false

            if (hands.size == 2) {
                val l = hands[0]; val r = hands[1]
                val leftPeace = isPeace(l)
                val rightPinch = isPinch(r)
                if (leftPeace && rightPinch) {
                    sling = true
                    newLine = Pair(l[8], r[4])
                    lastSlingAt = now
                    lastSlingLine = newLine
                    note("KETAPEL [${NAMES[portalFilter]}] - LEPAS CUBITAN UNTUK TEMBAK!", 400)
                } else if (isPinch(l) && rightPinch) {
                    newLine = Pair(l[4], r[4])
                    if (now - lastSwitch > 1000) {
                        lastSwitch = now
                        portalFilter = portalFilter % (NAMES.size - 1) + 1
                        note("PORTAL: ${NAMES[portalFilter]}", 1000)
                    }
                } else {
                    // portal = frame made by both hands (thumb + index tips)
                    newQuad = floatArrayOf(l[4].x, l[4].y, l[8].x, l[8].y, r[8].x, r[8].y, r[4].x, r[4].y)
                }
            } else if (hands.size == 1 && isOpenPalm(hands[0]) && screenFilter != 0) {
                screenFilter = 0
                note("RESET - filter layar dimatikan", 1200)
            }

            // tolerate brief detection drop-outs so the line doesn't flicker / misfire
            if (!sling && wasSling && now - lastSlingAt < 350) { sling = true; newLine = lastSlingLine }

            // slingshot released -> whole screen takes the portal filter
            if (wasSling && !sling) {
                screenFilter = portalFilter
                note("KETAPEL DITEMBAK! Layar: ${NAMES[screenFilter]}", 1500)
            }
            wasSling = sling

            quad = smoothQuad(newQuad)
            line = newLine
            handsPx = hands
        } catch (e: Throwable) {
            Log.e("RetroLens", "result error", e)
        }
    }

    private fun note(s: String, ms: Long) { msg = s; msgUntil = System.currentTimeMillis() + ms }

    // ---- gesture helpers (landmark indices from MediaPipe Hands) ----
    private fun d(a: PointF, b: PointF) = hypot(a.x - b.x, a.y - b.y)
    private fun size(h: List<PointF>) = d(h[0], h[9])
    private fun isPinch(h: List<PointF>) = d(h[4], h[8]) / size(h) < 0.4f
    private fun up(h: List<PointF>, tip: Int, pip: Int) = d(h[0], h[tip]) > d(h[0], h[pip]) * 1.15f
    private fun isPeace(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && !up(h, 16, 14) && !up(h, 20, 18)
    private fun isOpenPalm(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && up(h, 16, 14) && up(h, 20, 18)
}
