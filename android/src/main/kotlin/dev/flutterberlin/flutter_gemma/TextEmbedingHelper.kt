package dev.flutterberlin.flutter_gemma

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder.TextEmbedderOptions

class TextEmbedderHelper(private val context: Context) {

    private var textEmbedder: TextEmbedder? = null

    fun setup(): Boolean {
        return try {
            val baseOptions = BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath(MODEL_UNIVERSAL_SENTENCE_ENCODER_PATH)
                .build()

            val options = TextEmbedderOptions.builder()
                .setBaseOptions(baseOptions)
                .build()

            textEmbedder = TextEmbedder.createFromOptions(context, options)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TextEmbedder: ${e.message}")
            false
        }
    }

    fun getEmbedding(text: String): List<Double>? {
        if (textEmbedder == null && !setup()) return null

        val startTime = SystemClock.uptimeMillis()
        return try {
            val embedding = textEmbedder
                ?.embed(text)
                ?.embeddingResult()
                ?.embeddings()
                ?.firstOrNull()
                ?.floatEmbedding()

            val inferenceTime = SystemClock.uptimeMillis() - startTime
            Log.d(TAG, "Embedding inference time: $inferenceTime ms")

            embedding?.map { it.toDouble() }
        } catch (e: Exception) {
            Log.e(TAG, "Embedding failed: ${e.message}")
            null
        }
    }

    fun close() {
        textEmbedder?.close()
        textEmbedder = null
    }

    companion object {
        private const val TAG = "TextEmbedderHelper"
        private const val MODEL_UNIVERSAL_SENTENCE_ENCODER_PATH = "universal_sentence_encoder.tflite"
    }
}
