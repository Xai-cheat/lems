package com.retrolens.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.ComponentActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import java.util.concurrent.Executors
import kotlin.math.hypot

class MainActivity : ComponentActivity() {

    external fun applyFilters(bmp: Bitmap, screenFilter: Int, portalFilter: Int, quad: FloatArray?)

    companion object {
        init { System.loadLibrary("retrolens") }
        val NAMES = arrayOf("NONE", "CYBER_NEON", "SEPIA", "NEGATIVE")
    }

    private lateinit var imageView: ImageView
    private var landmarker: HandLandmarker? = null
    private val exec = Executors.newSingleThreadExecutor()

    // state
    private var screenFilter = 0
    private var portalFilter = 1
    private var lastSwitch = 0L
    private var wasSling = false
    private var msg = ""
    private var msgUntil = 0L
    private var lastFrame = System.nanoTime()
    private var fps = 0f

    private val green = Paint().apply { color = Color.rgb(60, 255, 60); strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true }
    private val dot = Paint().apply { color = Color.rgb(60, 255, 60); style = Paint.Style.FILL; isAntiAlias = true }
    private val text = Paint().apply { color = Color.rgb(60, 255, 60); textSize = 38f; isAntiAlias = true; typeface = Typeface.DEFAULT_BOLD }

    private val permLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imageView = ImageView(this).apply { scaleType = ImageView.ScaleType.FIT_CENTER; setBackgroundColor(Color.BLACK) }
        setContentView(FrameLayout(this).apply { addView(imageView) })

        try {
            landmarker = HandLandmarker.createFromOptions(
                this,
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
                    .setNumHands(2)
                    .setRunningMode(RunningMode.IMAGE)
                    .build()
            )
        } catch (e: Throwable) {
            setContentView(TextView(this).apply {
                setTextColor(Color.WHITE); setBackgroundColor(Color.BLACK); setPadding(40, 80, 40, 40)
                text = "Gagal load model:\n${e}"
            })
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            startCamera() else permLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(exec) { proxy -> processFrame(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(proxy: ImageProxy) {
        val raw = proxy.toBitmap()
        val m = Matrix().apply {
            postRotate(proxy.imageInfo.rotationDegrees.toFloat())
            postScale(-1f, 1f)  // mirror like a selfie view
        }
        val bmp = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            .copy(Bitmap.Config.ARGB_8888, true)
        proxy.close()

        val w = bmp.width.toFloat(); val h = bmp.height.toFloat()
        val result = landmarker!!.detect(BitmapImageBuilder(bmp).build())
        // hands as pixel-space points, sorted left-to-right on screen
        val hands = result.landmarks()
            .map { lm -> lm.map { PointF(it.x() * w, it.y() * h) } }
            .sortedBy { it[0].x }

        val now = System.currentTimeMillis()
        var quad: FloatArray? = null
        var line: Pair<PointF, PointF>? = null
        var sling = false

        if (hands.size == 2) {
            val l = hands[0]; val r = hands[1]
            val leftPeace = isPeace(l)
            val rightPinch = isPinch(r)
            if (leftPeace && rightPinch) {
                sling = true
                line = Pair(l[8], r[4])
                note("KETAPEL [${NAMES[portalFilter]}] - LEPAS CUBITAN UNTUK TEMBAK!", 300)
            } else if (isPinch(l) && rightPinch) {
                line = Pair(l[4], r[4])
                if (now - lastSwitch > 1000) {
                    lastSwitch = now
                    portalFilter = portalFilter % 3 + 1
                    note("PORTAL: ${NAMES[portalFilter]}", 1000)
                }
            } else {
                // portal = frame made by both hands (thumb + index tips)
                quad = floatArrayOf(l[4].x, l[4].y, l[8].x, l[8].y, r[8].x, r[8].y, r[4].x, r[4].y)
            }
        } else if (hands.size == 1 && isOpenPalm(hands[0]) && screenFilter != 0) {
            screenFilter = 0
            note("RESET - filter layar dimatikan", 1200)
        }

        // slingshot released -> whole screen takes the portal filter
        if (wasSling && !sling) {
            screenFilter = portalFilter
            note("KETAPEL DITEMBAK! Layar: ${NAMES[screenFilter]}", 1500)
        }
        wasSling = sling

        applyFilters(bmp, screenFilter, portalFilter, quad)

        // overlay
        val c = Canvas(bmp)
        quad?.let {
            val p = Path().apply { moveTo(it[0], it[1]); lineTo(it[2], it[3]); lineTo(it[4], it[5]); lineTo(it[6], it[7]); close() }
            c.drawPath(p, green)
            c.drawText("PORTAL: ${NAMES[portalFilter]}", it[2], minOf(it[3], it[5]) - 20, text)
        }
        line?.let { c.drawLine(it.first.x, it.first.y, it.second.x, it.second.y, green) }
        hands.forEach { hd -> intArrayOf(4, 8).forEach { i -> c.drawCircle(hd[i].x, hd[i].y, 10f, dot) } }

        val t = System.nanoTime(); fps = 0.9f * fps + 0.1f * (1e9f / (t - lastFrame)); lastFrame = t
        c.drawText("RETROLENS AR SYSTEM", 20f, 60f, text)
        c.drawText("Filter Layar: ${NAMES[screenFilter]}", 20f, 105f, text)
        c.drawText("FPS: %.1f".format(fps), 20f, 150f, text)
        if (now < msgUntil) c.drawText(msg, 20f, h - 120f, text)
        text.textSize = 28f
        c.drawText("2 Tangan = PORTAL | 2x Cubit = GANTI | Peace+Cubit = KETAPEL | 1 Telapak = RESET", 20f, h - 50f, text)
        text.textSize = 38f

        runOnUiThread { imageView.setImageBitmap(bmp) }
    }

    private fun note(s: String, ms: Long) { msg = s; msgUntil = System.currentTimeMillis() + ms }

    // ---- gesture helpers (landmark indices from MediaPipe Hands) ----
    private fun d(a: PointF, b: PointF) = hypot(a.x - b.x, a.y - b.y)
    private fun size(h: List<PointF>) = d(h[0], h[9])
    private fun isPinch(h: List<PointF>) = d(h[4], h[8]) / size(h) < 0.35f
    private fun up(h: List<PointF>, tip: Int, pip: Int) = d(h[0], h[tip]) > d(h[0], h[pip]) * 1.15f
    private fun isPeace(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && !up(h, 16, 14) && !up(h, 20, 18)
    private fun isOpenPalm(h: List<PointF>) = up(h, 8, 6) && up(h, 12, 10) && up(h, 16, 14) && up(h, 20, 18)
}
