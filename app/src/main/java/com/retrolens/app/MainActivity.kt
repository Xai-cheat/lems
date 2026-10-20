package com.retrolens.app

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
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
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------

/**
 * One-Euro filter: smooths hard when a point is still (no jitter) and follows quickly when it
 * moves fast (no lag). Much nicer than a fixed "move 45% of the way" blend.
 */
private class OneEuro(private val minCutoff: Float, private val beta: Float) {
    private var has = false
    private var xPrev = 0f
    private var dxPrev = 0f
    private var tPrev = 0L

    fun reset() { has = false }

    private fun alpha(cutoff: Float, dt: Float): Float {
        val tau = 1f / (2f * Math.PI.toFloat() * cutoff)
        return 1f / (1f + tau / dt)
    }

    fun filter(x: Float, tNanos: Long): Float {
        if (!has) {
            has = true; xPrev = x; dxPrev = 0f; tPrev = tNanos
            return x
        }
        val dt = ((tNanos - tPrev) / 1e9f).coerceIn(0.001f, 0.25f)
        tPrev = tNanos
        val dx = (x - xPrev) / dt
        dxPrev += alpha(1.0f, dt) * (dx - dxPrev)
        val cutoff = minCutoff + beta * abs(dxPrev)
        xPrev += alpha(cutoff, dt) * (x - xPrev)
        return xPrev
    }
}

/** Everything the drawing thread needs from the latest hand result, in one immutable snapshot. */
private class Overlay(
    val quad: FloatArray?,   // portal frame, 4 points
    val link: FloatArray?,   // line between two pinching hands (filter-switch gesture)
    val aim: FloatArray?,    // slingshot: indexTip(2), middleTip(2), pinchPoint(2), anchor(2)
    val charge: Float,       // 0..1
    val dots: FloatArray     // thumb + index tips of every hand
)

/** A circular "shockwave" that swaps the old screen filter for the new one. */
private class Reveal(
    val from: Int, val to: Int,
    val x: Float, val y: Float,
    val start: Long, val dur: Long
)

private enum class SlingState { IDLE, AIMING }

class MainActivity : ComponentActivity() {

    external fun applyFilters(
        bmp: Bitmap, screenFilter: Int, portalFilter: Int, quad: FloatArray?,
        prevFilter: Int, revealX: Float, revealY: Float, revealR: Float
    )

    companion object {
        init { System.loadLibrary("retrolens") }
        // filters the two-hand pinch cycles through (SEPIA removed)
        val CYCLE = intArrayOf(1, 3, 4, 5, 6, 7)
        val NAMES = arrayOf("NONE", "CYBER_NEON", "SEPIA", "NEGATIVE", "THERMAL", "NOIR", "MATRIX", "SUNSET")

        private const val TAG = "RetroLens"

        // ---- gesture tuning (all in one place) ----
        private const val PINCH_ON = 0.38f          // thumb-index gap / hand size to START a pinch
        private const val PINCH_OFF = 0.58f         // gap needed to END a pinch (hysteresis = no flicker)
        private const val POSE_CONFIRM = 3          // frames of "peace + pinch" before the sling starts
        private const val RELEASE_CONFIRM = 2       // frames of "pinch open" before we fire
        private const val LOST_GRACE_MS = 350L      // hands vanish while charged -> fire after this
        private const val PEACE_BROKEN_MS = 1000L   // peace sign dropped but pinch held -> cancel
        private const val PULL_MIN = 1.0f           // pull distance (in hand sizes) = 0% charge
        private const val PULL_MAX = 3.8f           // pull distance (in hand sizes) = 100% charge
        private const val MIN_FIRE_CHARGE = 0.20f   // below this the shot is cancelled
        private const val CYCLE_HOLD_MS = 350L      // both hands pinched this long -> next filter
        private const val CYCLE_REPEAT_MS = 1000L
        private const val AFTER_SLING_QUIET_MS = 1200L  // ignore palm-reset / cycling right after a shot
        private const val PALM_HOLD_MS = 500L       // open palm must be held this long to reset
        private const val QUAD_HOLD_MS = 150L       // keep the portal alive through a short drop-out
        private const val EASE_TAU_MS = 30f         // drawing-side smoothing time constant
        private const val DET_INTERVAL_MS = 33L
    }

    private lateinit var surfaceView: SurfaceView
    @Volatile private var surfaceOk = false
    private val blit = Paint().apply { isFilterBitmap = true }
    private val dst = RectF()
    @Volatile private var landmarker: HandLandmarker? = null
    @Volatile private var usingGpu = false
    @Volatile private var switching = false
    private var errView: TextView? = null
    private var analysisUse: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var rawBuf: Bitmap? = null
    private var listenerOn = false
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) { updateRotation() }
    }

    private val exec = Executors.newSingleThreadExecutor()     // camera frames -> drawing
    private val ioExec = Executors.newSingleThreadExecutor()   // saving photos

    @Volatile private var lensFront = true
    private val captureReq = AtomicBoolean(false)

    // reusable buffers (no per-frame allocation = no GC stutter)
    private val ring = arrayOfNulls<Bitmap>(3)
    private val ringCanvas = arrayOfNulls<Canvas>(3)
    private var ringIdx = 0
    private var detBmp: Bitmap? = null
    private var detCanvas: Canvas? = null
    private val detDst = Rect()
    private val xform = Matrix()
    private val portalPath = Path()

    // frame size, written by the drawing thread, read by the result callback
    @Volatile private var frameW = 0f
    @Volatile private var frameH = 0f

    // ---- state shared between the threads ----
    @Volatile private var screenFilter = 0
    @Volatile private var portalFilter = 1
    @Volatile private var overlay: Overlay? = null
    @Volatile private var reveal: Reveal? = null
    @Volatile private var msg = ""
    @Volatile private var msgUntil = 0L
    @Volatile private var flashStart = 0L
    @Volatile private var resetGen = 0

    // code log shown at the bottom of the screen: (language, code line)
    private val logLines = ArrayList<Pair<String, String>>()
    private fun logEv(lang: String, code: String) {
        synchronized(logLines) {
            logLines.add(Pair(lang, code))
            while (logLines.size > 4) logLines.removeAt(0)
        }
    }

    // ---- result-callback thread only ----
    private var seenGen = 0
    private val handFilters = Array(2) { Array(42) { OneEuro(1.5f, 0.02f) } }
    private var filterCount = -1
    private val pinched = BooleanArray(2)
    private var slingState = SlingState.IDLE
    private var poseFrames = 0
    private var peaceSlot = 0
    private var pinchSlot = 1
    private var peaceWrist = PointF()
    private var pinchWrist = PointF()
    private var releaseFrames = 0
    private var lastTrackedAt = 0L
    private var peaceBrokenSince = 0L
    private val chargeHist = FloatArray(5)
    private var chargeIdx = 0
    private var lastAim: FloatArray? = null
    private var lastCharge = 0f
    private var slingEndedAt = 0L
    private var bothPinchSince = 0L
    private var palmSince = 0L
    private var lastSwitch = 0L
    private var lastQuad: FloatArray? = null
    private var lastQuadAt = 0L
    private var hadQuad = false
    private var lastHandCount = -1
    private var lastHandLog = 0L

    // ---- drawing thread only ----
    private var lastTs = 0L
    private var lastDetSent = 0L
    private var lastDrawMs = 0L
    private var dispQuad: FloatArray? = null
    private var dispLink: FloatArray? = null
    private var dispAim: FloatArray? = null
    private var dispDots: FloatArray? = null
    private var dispCharge = 0f

    private val green = Paint().apply { color = Color.rgb(60, 255, 60); strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true }
    private val dot = Paint().apply { color = Color.rgb(60, 255, 60); style = Paint.Style.FILL; isAntiAlias = true }
    private val text = Paint().apply { color = Color.rgb(60, 255, 60); textSize = 38f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }
    private val mono = Paint().apply { typeface = Typeface.MONOSPACE; isAntiAlias = true }
    private val panelBg = Paint().apply { color = Color.argb(160, 0, 0, 0); style = Paint.Style.FILL }
    private val barBg = Paint().apply { color = Color.argb(140, 0, 0, 0); style = Paint.Style.FILL }
    private val flashPaint = Paint().apply { color = Color.WHITE; style = Paint.Style.FILL }
    private val aimLine = Paint().apply { style = Paint.Style.STROKE; isAntiAlias = true; strokeCap = Paint.Cap.ROUND }
    private val aimFill = Paint().apply { style = Paint.Style.FILL; isAntiAlias = true }
    private val ringPaint = Paint().apply { style = Paint.Style.STROKE; isAntiAlias = true; color = Color.WHITE }

    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera()
        else errView?.text = "Camera permission denied.\nEnable it in Settings > Apps > RetroLens AR."
    }
    private val storageLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) captureReq.set(true)
        else Toast.makeText(this, "Storage permission denied - can't save photos", Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        surfaceView = SurfaceView(this)
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) { surfaceOk = true }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}
            override fun surfaceDestroyed(holder: SurfaceHolder) { surfaceOk = false }
        })

        // tap = photo, double tap = flip camera
        val taps = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean { requestPhoto(); return true }
            override fun onDoubleTap(e: MotionEvent): Boolean { flipCamera(); return true }
        })
        surfaceView.setOnTouchListener { _, ev -> taps.onTouchEvent(ev); true }

        val root = FrameLayout(this)
        root.addView(surfaceView)
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
                    text = "Failed to load model:\n${e}"
                })
                return
            }
        }
        logEv("C++", "System.loadLibrary(\"retrolens\");")
        logEv("C#", "HandLandmarker.Create(${if (usingGpu) "GPU" else "CPU"});")
        note("Tap = photo | Double tap = flip camera", 3500)

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
        Log.e(TAG, "mediapipe error", e)
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
        // BUG FIX: this flag used to stay true forever, which silently turned hand
        // tracking (and therefore the slingshot) off after any GPU hiccup.
        switching = false
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider
                var selector = if (lensFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                if (!provider.hasCamera(selector)) {
                    // this phone has no camera facing that way - try the other one
                    lensFront = !lensFront
                    selector = if (lensFront) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                    if (!provider.hasCamera(selector)) {
                        errView?.text = "No usable camera found."
                        return@addListener
                    }
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 640))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(exec) { proxy -> processFrame(proxy) }
                analysisUse = analysis
                if (!listenerOn) {
                    listenerOn = true
                    (getSystemService(Context.DISPLAY_SERVICE) as DisplayManager).registerDisplayListener(displayListener, null)
                }
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, analysis)
            } catch (e: Throwable) {
                Log.e(TAG, "camera start failed", e)
                errView?.text = "Camera error: $e"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun flipCamera() {
        lensFront = !lensFront
        requestReset()
        note(if (lensFront) "Front camera" else "Back camera", 1000)
        logEv("C#", "Camera.Flip(${if (lensFront) "FRONT" else "BACK"});")
        startCamera()
    }

    private fun requestPhoto() {
        if (Build.VERSION.SDK_INT <= 28 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
        ) {
            storageLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            return
        }
        captureReq.set(true)
    }

    @Suppress("DEPRECATION")
    private fun savePhoto(bmp: Bitmap) {
        try {
            val name = "RetroLens_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                if (Build.VERSION.SDK_INT >= 29) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/RetroLens")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                } else {
                    val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "RetroLens")
                    dir.mkdirs()
                    put(MediaStore.Images.Media.DATA, File(dir, name).absolutePath)
                }
            }
            val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("MediaStore insert failed")
            contentResolver.openOutputStream(uri)?.use { out ->
                if (!bmp.compress(Bitmap.CompressFormat.JPEG, 95, out)) throw IllegalStateException("JPEG encode failed")
            } ?: throw IllegalStateException("Cannot open output stream")
            if (Build.VERSION.SDK_INT >= 29) {
                val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
                contentResolver.update(uri, done, null, null)
            }
            runOnUiThread { Toast.makeText(this, "Photo saved to Pictures/RetroLens", Toast.LENGTH_SHORT).show() }
        } catch (e: Throwable) {
            Log.e(TAG, "save photo failed", e)
            runOnUiThread { Toast.makeText(this, "Could not save photo: ${e.message}", Toast.LENGTH_LONG).show() }
        } finally {
            bmp.recycle()
        }
    }

    @Suppress("DEPRECATION")
    private fun updateRotation() {
        analysisUse?.targetRotation = windowManager.defaultDisplay.rotation
    }

    override fun onDestroy() {
        super.onDestroy()
        if (listenerOn) {
            listenerOn = false
            (getSystemService(Context.DISPLAY_SERVICE) as DisplayManager).unregisterDisplayListener(displayListener)
        }
        try { cameraProvider?.unbindAll() } catch (_: Throwable) {}
        val lm = landmarker
        landmarker = null
        exec.execute { try { lm?.close() } catch (_: Throwable) {} }
        exec.shutdown()
        ioExec.shutdown()
    }

    // copy the camera pixels into a reused bitmap (no new 1MB bitmap every frame)
    private fun grabFrame(proxy: ImageProxy): Bitmap {
        val iw = proxy.width
        val ih = proxy.height
        val plane = proxy.planes[0]
        val buf = plane.buffer
        if (plane.pixelStride == 4 && plane.rowStride == iw * 4 && buf.remaining() >= iw * ih * 4) {
            val rb = rawBuf?.takeIf { it.width == iw && it.height == ih }
                ?: Bitmap.createBitmap(iw, ih, Bitmap.Config.ARGB_8888).also { rawBuf = it }
            buf.rewind()
            rb.copyPixelsFromBuffer(buf)
            return rb
        }
        return proxy.toBitmap()
    }

    private fun nextBuffer(w: Int, h: Int): Bitmap {
        ringIdx = (ringIdx + 1) % ring.size
        var b = ring[ringIdx]
        if (b == null || b.width != w || b.height != h) {
            b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            ring[ringIdx] = b
            ringCanvas[ringIdx] = Canvas(b)
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

    // glide `cur` toward `tgt` in place (no allocation once the arrays exist)
    private fun easeTo(cur: FloatArray?, tgt: FloatArray?, k: Float): FloatArray? {
        if (tgt == null) return null
        if (cur == null || cur.size != tgt.size) return tgt.copyOf()
        for (i in tgt.indices) cur[i] += (tgt[i] - cur[i]) * k
        return cur
    }

    private fun chargeColor(c: Float): Int {
        val r: Int
        val g: Int
        if (c < 0.5f) { r = (60 + 195 * (c / 0.5f)).toInt(); g = 255 }
        else { r = 255; g = (255 * (1f - (c - 0.5f) / 0.5f)).toInt() }
        return Color.rgb(r.coerceIn(0, 255), g.coerceIn(0, 255), 60)
    }

    // ---------- drawing thread: runs on every camera frame, never waits for detection ----------
    private fun processFrame(proxy: ImageProxy) {
        try {
            val raw = grabFrame(proxy)
            val rot = proxy.imageInfo.rotationDegrees
            proxy.close()

            val swap = rot == 90 || rot == 270
            val ow = if (swap) raw.height else raw.width
            val oh = if (swap) raw.width else raw.height
            val bmp = nextBuffer(ow, oh)
            val c = ringCanvas[ringIdx]!!

            // rotate upright (+ mirror for the selfie camera), drawn straight into the reusable buffer
            xform.reset()
            xform.postTranslate(-raw.width / 2f, -raw.height / 2f)
            xform.postRotate(rot.toFloat())
            if (lensFront) xform.postScale(-1f, 1f)
            xform.postTranslate(ow / 2f, oh / 2f)
            c.drawBitmap(raw, xform, null)

            val w = bmp.width.toFloat(); val h = bmp.height.toFloat()
            if (w != frameW || h != frameH) {
                frameW = w; frameH = h
                overlay = null; reveal = null
                requestReset()
            }
            val k = minOf(w, h) / 480f   // UI scale, so everything looks the same on any resolution

            val nowMs = System.currentTimeMillis()
            val dt = (nowMs - lastDrawMs).coerceIn(1L, 100L)
            lastDrawMs = nowMs
            val ek = 1f - exp(-dt.toFloat() / EASE_TAU_MS)

            // hand the frame to MediaPipe (it runs in the background and skips frames if busy)
            val lm = landmarker
            if (lm != null && !switching && nowMs - lastDetSent >= DET_INTERVAL_MS) {
                lastDetSent = nowMs
                val dw = bmp.width / 2
                val dh = bmp.height / 2
                val db: Bitmap = detBmp?.takeIf { it.width == dw && it.height == dh }
                    ?: Bitmap.createBitmap(dw, dh, Bitmap.Config.ARGB_8888).also { detBmp = it; detCanvas = Canvas(it) }
                detDst.set(0, 0, dw, dh)
                detCanvas!!.drawBitmap(bmp, null, detDst, blit)
                try {
                    lm.detectAsync(BitmapImageBuilder(db).build(), nextTs())
                } catch (t: Throwable) {
                    Log.e(TAG, "detectAsync failed", t)
                    if (usingGpu && !switching) {
                        switching = true
                        runOnUiThread { switchToCpu() }
                    }
                }
            }
            // glide the overlay toward the newest detection (frame-rate independent)
            val ov = overlay
            dispQuad = easeTo(dispQuad, ov?.quad, ek)
            dispLink = easeTo(dispLink, ov?.link, ek)
            dispAim = easeTo(dispAim, ov?.aim, ek)
            dispDots = easeTo(dispDots, ov?.dots, minOf(1f, ek * 1.2f))
            dispCharge += ((ov?.charge ?: 0f) - dispCharge) * ek
            if (ov?.aim == null) dispCharge = 0f

            // shockwave reveal progress
            val rv = reveal
            var rvPrev = -1
            var rvR = -1f
            var rvX = 0f
            var rvY = 0f
            var rvP = 1f
            if (rv != null) {
                val p = ((nowMs - rv.start).toFloat() / rv.dur).coerceIn(0f, 1f)
                if (p < 1f) {
                    val maxR = max(max(hypot(rv.x, rv.y), hypot(w - rv.x, rv.y)),
                                   max(hypot(rv.x, h - rv.y), hypot(w - rv.x, h - rv.y)))
                    val e = 1f - (1f - p) * (1f - p) * (1f - p)   // ease-out
                    rvPrev = rv.from; rvX = rv.x; rvY = rv.y; rvP = p
                    rvR = maxR * e
                }
            }

            applyFilters(bmp, screenFilter, portalFilter, dispQuad, rvPrev, rvX, rvY, rvR)

            // photo: grab the filtered picture BEFORE any HUD is drawn on it
            if (captureReq.getAndSet(false)) {
                val snap = bmp.copy(Bitmap.Config.ARGB_8888, false)
                flashStart = nowMs
                if (snap != null) ioExec.execute { savePhoto(snap) }
            }

            // soft white flash (slingshot shot / photo)
            val fa = 1f - (nowMs - flashStart) / 220f
            if (fa > 0f && fa <= 1f) {
                flashPaint.alpha = (fa * 110).toInt().coerceIn(0, 255)
                c.drawRect(0f, 0f, w, h, flashPaint)
            }

            // glowing ring on the edge of the shockwave
            if (rvR > 0f) {
                ringPaint.strokeWidth = 5f * k
                ringPaint.alpha = ((1f - rvP) * 220).toInt().coerceIn(0, 255)
                c.drawCircle(rvX, rvY, rvR, ringPaint)
            }

            green.strokeWidth = 5f * k
            text.textSize = 34f * k

            dispQuad?.let {
                portalPath.rewind()
                portalPath.moveTo(it[0], it[1]); portalPath.lineTo(it[2], it[3])
                portalPath.lineTo(it[4], it[5]); portalPath.lineTo(it[6], it[7]); portalPath.close()
                c.drawPath(portalPath, green)
            }
            dispLink?.let { c.drawLine(it[0], it[1], it[2], it[3], green) }

            // ---- slingshot: elastic bands + charge bar ----
            val a = dispAim
            if (a != null) {
                val ch = dispCharge.coerceIn(0f, 1f)
                val col = chargeColor(ch)
                aimLine.color = col
                aimLine.strokeWidth = (4f + 5f * ch) * k
                c.drawLine(a[0], a[1], a[4], a[5], aimLine)     // index tip  -> pinch
                c.drawLine(a[2], a[3], a[4], a[5], aimLine)     // middle tip -> pinch
                aimFill.color = col
                c.drawCircle(a[4], a[5], (9f + 14f * ch) * k, aimFill)
                c.drawCircle(a[6], a[7], 7f * k, dot)

                val mx = (a[4] + a[6]) / 2f
                val my = (a[5] + a[7]) / 2f
                val bw = w * 0.36f
                val bh = h * 0.018f
                val bx = (mx - bw / 2f).coerceIn(4f, w - bw - 4f)
                val by = (my - bh - 40f * k).coerceIn(70f * k, h - 240f * k)
                c.drawRect(bx, by, bx + bw, by + bh, barBg)
                aimFill.color = col
                c.drawRect(bx, by, bx + bw * ch, by + bh, aimFill)
                // tick = minimum charge needed to fire
                c.drawRect(bx + bw * MIN_FIRE_CHARGE - 1.5f * k, by - 3f * k, bx + bw * MIN_FIRE_CHARGE + 1.5f * k, by + bh + 3f * k, dot)
                val pct = (ch * 100).toInt()
                val label = if (ch >= MIN_FIRE_CHARGE) "$pct% -> ${NAMES[portalFilter]}  RELEASE!" else "CHARGING $pct%"
                drawFit(c, label, bx, by - 8f * k, text, w - bx - 4f)
            }

            // finger dots
            val dd = dispDots
            if (dd != null) {
                for (i in 0 until dd.size / 4) {
                    c.drawCircle(dd[i * 4], dd[i * 4 + 1], 9f * k, dot)
                    c.drawCircle(dd[i * 4 + 2], dd[i * 4 + 3], 9f * k, dot)
                }
            }

            // code log panel at the bottom: what the app is doing right now
            val base = minOf(w, h) * 0.037f
            val lh = base * 1.5f
            val panelTop = h - (lh * 4f + 10f)
            c.drawRect(0f, panelTop, w, h, panelBg)
            val snapLog = synchronized(logLines) { ArrayList(logLines) }
            snapLog.forEachIndexed { i, entry ->
                val prefix = if (entry.first == "C#") "C#  " else "C++ "
                // measure at the base size first, then shrink to fit
                mono.textSize = base
                val full = mono.measureText(prefix + entry.second)
                val maxW = w - 24f
                mono.textSize = if (full > maxW) base * maxW / full else base
                val y = panelTop + 6f + lh * (i + 1) - lh * 0.25f
                val pw = mono.measureText(prefix)
                mono.color = if (entry.first == "C#") Color.rgb(197, 134, 192) else Color.rgb(86, 156, 214)
                c.drawText(prefix, 12f, y, mono)
                mono.color = Color.rgb(120, 255, 120)
                c.drawText(entry.second, 12f + pw, y, mono)
            }

            if (System.currentTimeMillis() < msgUntil) drawFit(c, msg, 20f * k, panelTop - 16f * k, text, w - 40f * k)

            // draw straight onto the screen surface (no hop through the UI thread)
            if (surfaceOk) {
                val sh = surfaceView.holder
                val sc = try { sh.lockHardwareCanvas() } catch (t: Throwable) { sh.lockCanvas() }
                if (sc != null) {
                    try {
                        sc.drawColor(Color.BLACK)
                        val vw = sc.width.toFloat()
                        val vh = sc.height.toFloat()
                        val sf = minOf(vw / w, vh / h)
                        val dw = w * sf
                        val dh = h * sf
                        dst.set((vw - dw) / 2f, (vh - dh) / 2f, (vw + dw) / 2f, (vh + dh) / 2f)
                        sc.drawBitmap(bmp, null, dst, blit)
                    } finally {
                        sh.unlockCanvasAndPost(sc)
                    }
                }
            }
        } catch (e: Throwable) {
            runCatching { proxy.close() }
            Log.e(TAG, "frame error", e)
            if (!switching) runOnUiThread { errView?.text = "ERR: $e" }
        }
    }

    // ---------- MediaPipe result callback: decides gestures ----------
    private fun requestReset() { resetGen++ }

    private fun resetGestureState() {
        slingState = SlingState.IDLE
        poseFrames = 0; releaseFrames = 0; peaceBrokenSince = 0L
        bothPinchSince = 0L; palmSince = 0L
        pinched.fill(false)
        chargeHist.fill(0f)
        lastAim = null; lastCharge = 0f
        lastQuad = null; hadQuad = false
        filterCount = -1
    }

    /** Hands as smoothed pixel-space points, sorted left-to-right on screen. */
    private fun buildHands(result: HandLandmarkerResult, w: Float, h: Float, nowN: Long): List<List<PointF>> {
        val lms = result.landmarks()
        val order = lms.indices.sortedBy { lms[it][0].x() }.take(2)
        if (order.size != filterCount) {
            // the left/right slots changed meaning, so start the smoothing fresh
            filterCount = order.size
            for (slot in handFilters) for (f in slot) f.reset()
        }
        return order.mapIndexed { slot, idx ->
            val lm = lms[idx]
            val fl = handFilters[slot]
            List(minOf(lm.size, 21)) { i ->
                PointF(
                    fl[i * 2].filter(lm[i].x() * 1000f, nowN) * (w / 1000f),
                    fl[i * 2 + 1].filter(lm[i].y() * 1000f, nowN) * (h / 1000f)
                )
            }
        }
    }
    private fun startReveal(from: Int, to: Int, origin: PointF, power: Float, now: Long) {
        val dur = (700f - 300f * power).toLong()
        reveal = Reveal(from, to, origin.x, origin.y, now, dur)
    }

    /** Ends the slingshot. Fires (changes the screen filter) only if it was charged enough. */
    private fun endSling(fire: Boolean, now: Long, w: Float, h: Float) {
        val peak = chargeHist.maxOrNull() ?: 0f
        val a = lastAim
        val origin = if (a != null) PointF(a[6], a[7]) else PointF(w / 2f, h / 2f)
        slingState = SlingState.IDLE
        poseFrames = 0; releaseFrames = 0; peaceBrokenSince = 0L
        lastAim = null; lastCharge = 0f
        slingEndedAt = now

        if (fire && peak >= MIN_FIRE_CHARGE) {
            val prev = screenFilter
            screenFilter = portalFilter
            flashStart = now
            if (prev == screenFilter) {
                note("Screen is already ${NAMES[screenFilter]}", 1200)
            } else {
                startReveal(prev, screenFilter, origin, peak, now)
                note("SLINGSHOT FIRED! Screen: ${NAMES[screenFilter]}", 1500)
                logEv("C#", "Screen.Filter = ${NAMES[screenFilter]}; // BANG")
                logEv("C++", "applyFilters(bmp, ${NAMES[screenFilter]});")
            }
        } else {
            note("Pull back farther to charge the slingshot", 1200)
            logEv("C#", "Sling.Cancel(); // charge ${(peak * 100).toInt()}% < ${(MIN_FIRE_CHARGE * 100).toInt()}%")
        }
    }

    private fun onResult(result: HandLandmarkerResult) {
        try {
            val w = frameW; val h = frameH
            if (w == 0f || h == 0f) return
            if (seenGen != resetGen) { seenGen = resetGen; resetGestureState() }

            val nowN = System.nanoTime()
            val now = System.currentTimeMillis()
            val hands = buildHands(result, w, h, nowN)

            if (hands.size != lastHandCount && now - lastHandLog > 700) {
                lastHandCount = hands.size
                lastHandLog = now
                logEv("C#", "hands.Count = ${hands.size};")
            }

            // pinch state per hand slot, with hysteresis so it doesn't flicker around the threshold
            for (i in 0..1) {
                if (i < hands.size) {
                    val r = pinchRatio(hands[i])
                    pinched[i] = if (pinched[i]) r < PINCH_OFF else r < PINCH_ON
                } else {
                    pinched[i] = false
                }
            }

            var newQuad: FloatArray? = null
            var link: FloatArray? = null
            var aim: FloatArray? = null
            var charge = 0f

            if (slingState == SlingState.AIMING) {
                // ================= slingshot is being pulled =================
                if (hands.size == 2) {
                    lastTrackedAt = now
                    // keep the roles even if the hands cross: match each hand to where it was last frame
                    val straight = d(hands[0][0], peaceWrist) + d(hands[1][0], pinchWrist)
                    val crossed = d(hands[1][0], peaceWrist) + d(hands[0][0], pinchWrist)
                    if (crossed < straight) { peaceSlot = 1; pinchSlot = 0 } else { peaceSlot = 0; pinchSlot = 1 }
                    peaceWrist = hands[peaceSlot][0]; pinchWrist = hands[pinchSlot][0]
                    val peaceH = hands[peaceSlot]
                    val pinchH = hands[pinchSlot]
                    val from = mid(peaceH[8], peaceH[12])       // fork of the peace sign
                    val to = mid(pinchH[4], pinchH[8])          // the pinch point
                    // pull distance is measured in hand sizes, so it works at any distance from the camera
                    val pull = d(from, to) / max(size(pinchH), 1f)
                    charge = ((pull - PULL_MIN) / (PULL_MAX - PULL_MIN)).coerceIn(0f, 1f)
                    chargeHist[chargeIdx] = charge
                    chargeIdx = (chargeIdx + 1) % chargeHist.size
                    aim = floatArrayOf(peaceH[8].x, peaceH[8].y, peaceH[12].x, peaceH[12].y, to.x, to.y, from.x, from.y)
                    lastAim = aim
                    lastCharge = charge

                    if (pinched[pinchSlot]) releaseFrames = 0 else releaseFrames++
                    if (isPeaceLoose(peaceH)) peaceBrokenSince = 0L
                    else if (peaceBrokenSince == 0L) peaceBrokenSince = now

                    if (releaseFrames >= RELEASE_CONFIRM) endSling(true, now, w, h)               // pinch opened = FIRE
                    else if (peaceBrokenSince != 0L && now - peaceBrokenSince > PEACE_BROKEN_MS) endSling(false, now, w, h)
                } else {
                    // a hand dropped out of tracking: hold the last aim for a moment, then resolve
                    aim = lastAim
                    charge = lastCharge
                    if (now - lastTrackedAt > LOST_GRACE_MS) endSling(true, now, w, h)
                }
                if (slingState != SlingState.AIMING) { aim = null; charge = 0f }
            } else {
                // ================= normal gestures =================
                var peaceIdx = -1
                var pinchIdx = -1
                if (hands.size == 2) {
                    if (isPeace(hands[0]) && pinched[1]) { peaceIdx = 0; pinchIdx = 1 }
                    else if (isPeace(hands[1]) && pinched[0]) { peaceIdx = 1; pinchIdx = 0 }
                }
                poseFrames = if (peaceIdx >= 0) poseFrames + 1 else 0

                if (peaceIdx >= 0) {
                    bothPinchSince = 0L
                    palmSince = 0L
                    if (poseFrames >= POSE_CONFIRM) {
                        slingState = SlingState.AIMING
                        peaceSlot = peaceIdx; pinchSlot = pinchIdx
                        peaceWrist = hands[peaceIdx][0]; pinchWrist = hands[pinchIdx][0]
                        releaseFrames = 0; peaceBrokenSince = 0L
                        lastTrackedAt = now
                        chargeHist.fill(0f)
                        logEv("C#", "if (Peace && Pinch) Sling.Aim();")
                    }
                } else if (hands.size == 2) {
                    palmSince = 0L
                    val a = hands[0]; val b = hands[1]
                    if (pinched[0] && pinched[1]) {
                        // both hands pinching -> next portal filter (must be HELD, so it can't trigger by accident)
                        val ma = mid(a[4], a[8]); val mb = mid(b[4], b[8])
                        link = floatArrayOf(ma.x, ma.y, mb.x, mb.y)
                        if (bothPinchSince == 0L) bothPinchSince = now
                        if (now - bothPinchSince > CYCLE_HOLD_MS && now - lastSwitch > CYCLE_REPEAT_MS &&
                            now - slingEndedAt > AFTER_SLING_QUIET_MS) {
                            lastSwitch = now
                            portalFilter = CYCLE[(CYCLE.indexOf(portalFilter) + 1) % CYCLE.size]
                            note("PORTAL: ${NAMES[portalFilter]}", 1000)
                            logEv("C#", "Portal.Filter = ${NAMES[portalFilter]};")
                        }
                    } else {
                        bothPinchSince = 0L
                        // portal = frame made by both hands (thumb + index tips)
                        newQuad = floatArrayOf(a[4].x, a[4].y, a[8].x, a[8].y, b[8].x, b[8].y, b[4].x, b[4].y)
                    }
                } else {
                    bothPinchSince = 0L
                    // one open palm, held, resets the screen filter
                    if (hands.size == 1 && isOpenPalm(hands[0]) && screenFilter != 0 &&
                        now - slingEndedAt > AFTER_SLING_QUIET_MS) {
                        if (palmSince == 0L) palmSince = now
                        if (now - palmSince > PALM_HOLD_MS) {
                            val prev = screenFilter
                            screenFilter = 0
                            palmSince = 0L
                            startReveal(prev, 0, hands[0][9], 0.5f, now)
                            note("RESET - screen filter off", 1200)
                            logEv("C#", "Screen.Filter = NONE; // palm reset")
                        }
                    } else {
                        palmSince = 0L
                    }
                }
            }

            // keep the portal alive through a very short tracking drop-out (no flicker)
            if (newQuad != null) {
                lastQuad = newQuad; lastQuadAt = now
            } else if (hands.size < 2 && slingState == SlingState.IDLE && lastQuad != null && now - lastQuadAt < QUAD_HOLD_MS) {
                newQuad = lastQuad
            }
            if (newQuad != null && !hadQuad) logEv("C++", "drawQuad(thumbL,idxL,idxR,thumbR);")
            if (newQuad == null && hadQuad) logEv("C#", "Portal.Close();")
            hadQuad = newQuad != null

            val dots = FloatArray(hands.size * 4)
            hands.forEachIndexed { i, hd ->
                dots[i * 4] = hd[4].x; dots[i * 4 + 1] = hd[4].y
                dots[i * 4 + 2] = hd[8].x; dots[i * 4 + 3] = hd[8].y
            }
            overlay = Overlay(newQuad, link, aim, charge, dots)
        } catch (e: Throwable) {
            Log.e(TAG, "result error", e)
        }
    }

    private fun note(s: String, ms: Long) { msg = s; msgUntil = System.currentTimeMillis() + ms }

    // ---- gesture helpers (landmark indices from MediaPipe Hands) ----
    private fun d(a: PointF, b: PointF) = hypot(a.x - b.x, a.y - b.y)
    private fun mid(a: PointF, b: PointF) = PointF((a.x + b.x) / 2f, (a.y + b.y) / 2f)
    private fun size(h: List<PointF>) = d(h[0], h[9])
    private fun pinchRatio(h: List<PointF>) = d(h[4], h[8]) / max(size(h), 1f)
    private fun up(h: List<PointF>, tip: Int, pip: Int, k: Float = 1.15f) = d(h[0], h[tip]) > d(h[0], h[pip]) * k
    private fun isPeace(h: List<PointF>) = up(h, 8, 6, 1.1f) && up(h, 12, 10, 1.1f) && !up(h, 16, 14, 1.25f) && !up(h, 20, 18, 1.25f)
    private fun isPeaceLoose(h: List<PointF>) = up(h, 8, 6, 1.0f) && up(h, 12, 10, 1.0f)
    private fun isOpenPalm(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && up(h, 16, 14) && up(h, 20, 18)
}

    
            
