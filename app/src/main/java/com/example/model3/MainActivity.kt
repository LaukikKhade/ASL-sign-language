package com.example.model3

import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage

import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var resultText: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var translator: TFLiteTranslator

    private lateinit var faceLandmarker: FaceLandmarker
    private lateinit var handLandmarker: HandLandmarker
    private lateinit var poseLandmarker: PoseLandmarker

    private val frameBuffer = mutableListOf<Array<FloatArray>>()
    private val MAX_FRAMES = 10// Reduced for faster, snappier predictions
    private var frameCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultText = findViewById(R.id.tv_result)

        translator = TFLiteTranslator(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        setupModels()
        startCamera()
    }

    private fun setupModels() {
        val faceBase = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
        val faceOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(faceBase)
            .setRunningMode(RunningMode.IMAGE)
            .build()
        faceLandmarker = FaceLandmarker.createFromOptions(this, faceOptions)

        val handBase = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(handBase)
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(2)
            .build()
        handLandmarker = HandLandmarker.createFromOptions(this, handOptions)

        val poseBase = BaseOptions.builder().setModelAssetPath("pose_landmarker.task").build()
        val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(poseBase)
            .setRunningMode(RunningMode.IMAGE)
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(this, poseOptions)
    }

    private fun startCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            val analyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                val bitmapBuffer = imageProxy.toBitmap()

                val matrix = Matrix().apply {
                    postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                    postScale(-1f, 1f, bitmapBuffer.width / 2f, bitmapBuffer.height / 2f)
                }

                val rotatedBitmap = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true
                )

                val mpImage: MPImage = BitmapImageBuilder(rotatedBitmap).build()
                processFrame(mpImage)
                imageProxy.close()
            }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, analyzer)

        }, ContextCompat.getMainExecutor(this))
    }

    // Listens for you to click "Allow" on the camera popup!
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun processFrame(image: MPImage) {
        frameCount++
        // Check every 2nd frame instead of 3rd for smoother tracking
        if (frameCount % 2 != 0) return

        val face = faceLandmarker.detect(image).faceLandmarks().firstOrNull()
        val pose = poseLandmarker.detect(image).landmarks().firstOrNull()
        val handResult = handLandmarker.detect(image)

        var leftHand: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? =
            null
        var rightHand: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>? =
            null

        val hands = handResult.landmarks()
        val handedness = handResult.handednesses()

        for (i in hands.indices) {
            val label = handedness[i][0].categoryName()
            if (label == "Left") leftHand = hands[i]
            if (label == "Right") rightHand = hands[i]
        }

        val frame = Array(543) { FloatArray(3) }
        var index = 0

        fun fill(
            list: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>?,
            size: Int
        ) {
            if (list != null) {
                for (i in 0 until size) {
                    frame[index][0] = list[i].x()
                    frame[index][1] = list[i].y()
                    frame[index][2] = list[i].z()
                    index++
                }
            } else {
                repeat(size) {
                    frame[index][0] = Float.NaN
                    frame[index][1] = Float.NaN
                    frame[index][2] = Float.NaN
                    index++
                }
            }
        }

        fill(face, 468)
        fill(leftHand, 21)
        fill(pose, 33)
        fill(rightHand, 21)

        // Only record if the camera actually sees a hand
        // ALWAYS add the frame so the timeline keeps moving smoothly
        // ALWAYS add the frame so the timeline keeps moving smoothly
        frameBuffer.add(frame)

        if (frameBuffer.size > MAX_FRAMES) frameBuffer.removeAt(0)

        // Only predict if the buffer is full AND a hand is currently visible
        if (frameBuffer.size == MAX_FRAMES && (leftHand != null || rightHand != null)) {

            val (label, confidence) = translator.predict(frameBuffer.toTypedArray())

            Log.d("ASL_Predict", "Detected Sign: $label | Confidence: ${confidence * 100}%")

            // Let's raise the threshold back to 50% so it only triggers when sure
            if (confidence > 0.30f) {
                runOnUiThread {
                    resultText.text = "$label  (${String.format("%.1f", confidence * 100)}%)"
                }

                // THE FIX: Clear the buffer after a successful, confident prediction!
                // This acts as a reset so it starts collecting a fresh 30 frames
                // for your next sign instead of blending them together.
                frameBuffer.clear()
            }
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        translator.close()
    }
}