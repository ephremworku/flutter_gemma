import Flutter
import UIKit
import MediaPipeTasksText


@available(iOS 13.0, *)
public class FlutterGemmaPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
      let platformService = PlatformServiceImpl()
      PlatformServiceSetup.setUp(binaryMessenger: registrar.messenger(), api: platformService)

      let eventChannel = FlutterEventChannel(
        name: "flutter_gemma_stream", binaryMessenger: registrar.messenger())
      eventChannel.setStreamHandler(platformService)
  }
}

class PlatformServiceImpl : NSObject, PlatformService, FlutterStreamHandler {
    private var eventSink: FlutterEventSink?
    private var model: InferenceModel?
    private var session: InferenceSession?
    private var textEmbedderService: TextEmbedderService?

    func createModel(
        maxTokens: Int64,
        modelPath: String,
        loraRanks: [Int64]?,
        preferredBackend: PreferredBackend?,
        completion: @escaping (Result<Void, any Error>) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                self.model = try InferenceModel(
                    modelPath: modelPath,
                    maxTokens: Int(maxTokens),
                    supportedLoraRanks: loraRanks?.map(Int.init)
                )
                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    func closeModel(completion: @escaping (Result<Void, any Error>) -> Void) {
        model = nil
        completion(.success(()))
    }

    func createSession(
        temperature: Double,
        randomSeed: Int64,
        topK: Int64,
        topP: Double?,
        loraPath: String?,
        completion: @escaping (Result<Void, any Error>) -> Void
    ) {
        guard let inference = model?.inference else {
            completion(.failure(PigeonError(code: "Inference model not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let newSession = try InferenceSession(
                    inference: inference,
                    temperature: Float(temperature),
                    randomSeed: Int(randomSeed),
                    topK: Int(topK),
                    topP: topP,
                    loraPath: loraPath
                )
                DispatchQueue.main.async {
                    self.session = newSession
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    func sizeInTokens(prompt: String, completion: @escaping (Result<Int64, any Error>) -> Void) {
        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let tokenCount = try session.sizeInTokens(prompt: prompt)
                DispatchQueue.main.async { completion(.success(Int64(tokenCount))) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    func addQueryChunk(prompt: String, completion: @escaping (Result<Void, any Error>) -> Void) {
        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try session.addQueryChunk(prompt: prompt)
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    func closeSession(completion: @escaping (Result<Void, any Error>) -> Void) {
        session = nil
        completion(.success(()))
    }

    func generateResponse(completion: @escaping (Result<String, any Error>) -> Void) {
        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let response = try session.generateResponse()
                DispatchQueue.main.async { completion(.success(response)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    func generateResponseAsync(completion: @escaping (Result<Void, any Error>) -> Void) {
        guard let session = session, let eventSink = eventSink else {
            completion(.failure(PigeonError(code: "Session or eventSink not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let stream = try session.generateResponseAsync()
                Task.detached { [weak self] in
                    guard let self = self else { return }
                    do {
                        for try await token in stream {
                            DispatchQueue.main.async {
                                eventSink(["partialResult": token, "done": false])
                            }
                        }
                        DispatchQueue.main.async {
                            eventSink(FlutterEndOfEventStream)
                        }
                    } catch {
                        DispatchQueue.main.async {
                            eventSink(FlutterError(code: "ERROR", message: error.localizedDescription, details: nil))
                        }
                    }
                }
                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }

func getEmbeddingOfText(text: String, completion: @escaping (Result<[Double], any Error>) -> Void) {
    // Initialize the TextEmbedderService if not already initialized
    if textEmbedderService == nil {
        let modelPath = Bundle(for: type(of: self)).path(forResource: "text_embedder_model", ofType: "tflite")
        print("Model path: \(String(describing: modelPath))")
        textEmbedderService = TextEmbedderService(modelPath: modelPath)
    }

    // Get the embedding for the provided text
    if let embeddingResult = textEmbedderService?.embed(text: text),
       let firstEmbedding = embeddingResult.embeddings.first?.floatEmbedding {

        // Convert the embeddings from Float to Double
        let embeddingsAsDoubles = firstEmbedding.map { Double(truncating: $0) }
        completion(.success(embeddingsAsDoubles))
    } else {
        completion(.failure(PigeonError(
            code: "TEXT_EMBEDDING_FAILED \(String(describing: Bundle.main.path(forResource: "text_embedder_model", ofType: "tflite")))",
            message: "Failed to embed text",
            details: nil
        )))
    }
}



}


class TextEmbedderService {
    var textEmbedder: TextEmbedder?

    // Initialize with the path to the model
    init(modelPath: String?) {
        guard let modelPath = modelPath else { return }
        do {
            self.textEmbedder = try TextEmbedder(modelPath: modelPath)
        } catch {
            print("Failed to load TextEmbedder model: \(error)")
        }
    }

    // Embed the given text and return the embeddings
    func embed(text: String) -> EmbeddingResult? {
        guard let textEmbedder = textEmbedder else { return nil }
        return try? textEmbedder.embed(text: text).embeddingResult
    }
}
