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
    private val dot = Paint().apply { color = Color.rgb(60, 255, 60); style = Paint.Style.FILL; isAntiAlias = true }
    private val text = Paint().apply { color = Color.rgb(60, 255, 60); textSize = 38f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }
    private val mono = Paint().apply { typeface = Typeface.MONOSPACE; isAntiAlias = true }
    private val panelBg = Paint().apply { color = Color.argb(160, 0, 0, 0); style = Paint.Style.FILL }
    private val barBg = Paint().apply { color = Color.argb(140, 0, 0, 0); style = Paint.Style.FILL }
    private val flashPaint = Paint().apply { color = Color.argb(110, 255, 255, 255); style = Paint.Style.FILL }

    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imageView = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(Color.BLACK) }
        val root = FrameLayout(this)
        root.addView(imageView)
        errView = TextView(this).apply { setTextColor(Color.RED); textSize = 13f; setPadding(20, 140, 20, 0) }
        root.addView(errView)
        setContentView(root)

        // try the GPU first (much faster); fall back to CPU if this phone can't do it
        landmarker = try {
            usingGpu = true
            buildLandmarker(true)
        } catch (t: Throwable) {
            usingGpu = false
            try {
                buildLandmarker(false)
            } catch (e: Throwable) {
                setContentView(TextView(this).apply {
                    setTextColor(Color.WHITE); setBackgroundColor(Color.BLACK); setPadding(40, 80, 40, 40)
                    text = "Gagal load model:\n${e}"
                })
                return
            }
        }
        logEv("C++", "System.loadLibrary(\"retrolens\");")
        logEv("C#", "HandLandmarker.Create(${if (usingGpu) "GPU" else "CPU"});")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera() else permLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun buildLandmarker(gpu: Boolean): HandLandmarker {
        val base = BaseOptions.builder().setModelAssetPath("hand_landmarker.task")
        if (gpu) base.setDelegate(Delegate.GPU)
        return HandLandmarker.createFromOptions(
            this,
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base.build())
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.4f)
                .setMinHandPresenceConfidence(0.4f)
                .setMinTrackingConfidence(0.4f)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result: HandLandmarkerResult, _: MPImage -> onResult(result) }
                .setErrorListener { e: RuntimeException -> onMpError(e) }
                .build()
        )
    }

    private fun onMpError(e: RuntimeException) {
        Log.e("RetroLens", "mediapipe error", e)
        if (usingGpu && !switching) {
            switching = true
            runOnUiThread { switchToCpu() }
        } else {
            runOnUiThread { errView?.text = "ERR: $e" }
        }
    }

    private fun switchToCpu() {
        val old = landmarker
        landmarker = null
        try { old?.close() } catch (_: Throwable) {}
        usingGpu = false
        landmarker = try {
            buildLandmarker(false)
        } catch (t: Throwable) {
            errView?.text = "ERR: $t"
            null
        }
        logEv("C#", "HandLandmarker.Create(CPU); // fallback")
        errView?.text = ""
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(480, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(exec) { proxy -> processFrame(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun nextBuffer(w: Int, h: Int): Bitmap {
        ringIdx = (ringIdx + 1) % ring.size
        var b = ring[ringIdx]
        if (b == null || b.width != w || b.height != h) {
            b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            ring[ringIdx] = b
        }
        return b!!
    }

    private fun nextTs(): Long {
        val t = System.nanoTime() / 1_000_000
        lastTs = if (t > lastTs) t else lastTs + 1
        return lastTs
    }

    // draw text, shrinking it if it would run off the right edge
    private fun drawFit(c: Canvas, s: String, x: Float, y: Float, p: Paint, maxW: Float) {
        val old = p.textSize
        val wid = p.measureText(s)
        if (wid > maxW && maxW > 0f) p.textSize = old * maxW / wid
        c.drawText(s, x, y, p)
        p.textSize = old
    }

    // ---------- drawing thread: runs on every camera frame, never waits for detection ----------
    private fun processFrame(proxy: ImageProxy) {
        try {
            val raw = proxy.toBitmap()
            val rot = proxy.imageInfo.rotationDegrees
            proxy.close()

            val swap = rot == 90 || rot == 270
            val ow = if (swap) raw.height else raw.width
            val oh = if (swap) raw.width else raw.height
            val bmp = nextBuffer(ow, oh)

            // rotate upright + mirror (selfie view), drawn straight into the reusable buffer
            val m = Matrix().apply {
                postTranslate(-raw.width / 2f, -raw.height / 2f)
                postRotate(rot.toFloat())
                postScale(-1f, 1f)
                postTranslate(ow / 2f, oh / 2f)
            }
            val c = Canvas(bmp)
            c.drawBitmap(raw, m, null)

            val w = bmp.width.toFloat(); val h = bmp.height.toFloat()
            frameW = w; frameH = h

            // hand the frame to MediaPipe (it runs in the background and skips frames if busy)
            val lm = landmarker
            val nowMs = System.currentTimeMillis()
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
