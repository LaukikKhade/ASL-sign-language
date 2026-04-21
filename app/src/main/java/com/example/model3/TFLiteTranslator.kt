package com.example.model3

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteTranslator(private val context: Context) {

    private val interpreter: Interpreter
    private val labels = mutableListOf<String>()

    init {
        interpreter = Interpreter(loadModelFile(context, "model.tflite"))
        loadLabels()
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels() {
        val jsonString = context.assets.open("sign_to_prediction_index_map.json").bufferedReader().use { it.readText() }
        val jsonObject = JSONObject(jsonString)

        val tempMap = mutableMapOf<Int, String>()
        jsonObject.keys().forEach { signName ->
            val index = jsonObject.getInt(signName)
            tempMap[index] = signName
        }

        for (i in 0 until 250) {
            labels.add(tempMap[i] ?: "Unknown")
        }
    }

    fun predict(inputFrames: Array<Array<FloatArray>>): Pair<String, Float> {
        val frames = inputFrames.size

        // Model input shape: [1, frames, 543, 3]
        val input = Array(1) { Array(frames) { Array(543) { FloatArray(3) } } }

        for (i in 0 until frames) {
            for (j in 0 until 543) {
                for (k in 0 until 3) {
                    input[0][i][j][k] = inputFrames[i][j][k]
                }
            }
        }

        // Output shape: [250] (a flat 1D array)
        val output = FloatArray(250)
        interpreter.run(input, output)

        // --- APPLY SOFTMAX TO CONVERT LOGITS TO PROBABILITIES ---
        var maxVal = output[0]
        for (value in output) {
            if (value > maxVal) maxVal = value
        }

        var sumExp = 0.0f
        for (i in output.indices) {
            output[i] = kotlin.math.exp((output[i] - maxVal).toDouble()).toFloat()
            sumExp += output[i]
        }

        var maxIndex = 0
        var maxProb = 0.0f
        for (i in output.indices) {
            output[i] /= sumExp // Normalize to 0.0 - 1.0

            if (output[i] > maxProb) {
                maxProb = output[i]
                maxIndex = i
            }
        }

        val label = labels.getOrElse(maxIndex) { "Unknown" }
        return Pair(label, maxProb)
    }

    fun close() {
        interpreter.close()
    }
}