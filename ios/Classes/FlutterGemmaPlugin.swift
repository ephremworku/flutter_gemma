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

    private var textEmbedder: TextEmbedder?

    func getEmbeddingOfText(text: String, completion: @escaping (Result<[Double], Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // Lazy-load the embedder if needed
                if self.textEmbedder == nil {
                    let bundle = Bundle(for: Self.self)
                    let modelPath = bundle.path(forResource: "text_embedder_model", ofType: "tflite")
                    guard let modelPath = modelPath else {
                        throw PigeonError(code: "MODEL_NOT_FOUND", message: "text_embedder_model.tflite not found in bundle", details: nil)
                    }

                    let options = TextEmbedderOptions()
                    options.baseOptions.modelAssetPath = modelPath
                    options.quantize = true

                    self.textEmbedder = try TextEmbedder(options: options)
                }

                guard let embeddingResult = try self.textEmbedder?.embed(text: text) else {
                    throw PigeonError(code: "EMBEDDING_FAILED", message: "Failed to embed text", details: nil)
                }

                let embedding = embeddingResult.embeddings.first?.values.map { Double($0) } ?? []

                DispatchQueue.main.async {
                    completion(.success(embedding))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }

}
