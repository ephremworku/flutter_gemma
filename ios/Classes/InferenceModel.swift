//import Foundation
//import MediaPipeTasksGenAI
//import MediaPipeTasksGenAIC
//import MediaPipeTasksText
//
//
//struct InferenceModel {
//    private(set) var inference: LlmInference
//
//    init(modelPath: String, maxTokens: Int, supportedLoraRanks: [Int]?) throws {
//        let fileManager = FileManager.default
//
//        guard let documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
//            throw NSError(domain: "InferenceModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Document directory not found"])
//        }
//
//        let fileName = (modelPath as NSString).lastPathComponent
//
//        let resolvedPath = documentDirectory.appendingPathComponent(fileName).path
//
//        let llmOptions = LlmInference.Options(modelPath: resolvedPath)
//        llmOptions.maxTokens = maxTokens
//        llmOptions.waitForWeightUploads = true
//        if let supportedLoraRanks = supportedLoraRanks {
//          llmOptions.supportedLoraRanks = supportedLoraRanks
//        }
//        self.inference = try LlmInference(options: llmOptions)
//    }
//}
//
//final class InferenceSession {
//    private let session: LlmInference.Session
//
//    init(inference: LlmInference, temperature: Float, randomSeed: Int, topK: Int, topP: Double? = nil, loraPath: String? = nil) throws {
//        let options = LlmInference.Session.Options()
//        options.temperature = temperature
//        options.randomSeed = randomSeed
//        options.topk = topK
//        if let topP = topP {
//            options.topp = Float(topP)
//        }
//        if let loraPath = loraPath {
//            options.loraPath = loraPath
//        }
//        self.session = try LlmInference.Session(llmInference: inference, options: options)
//    }
//
//
//    func sizeInTokens(prompt: String) throws -> Int {
//        return try session.sizeInTokens(text: prompt)
//    }
//
//    func addQueryChunk(prompt: String) throws {
//        try session.addQueryChunk(inputText: prompt)
//    }
//
//    func generateResponse(prompt: String? = nil) throws -> String {
//        if let prompt = prompt {
//            try session.addQueryChunk(inputText: prompt)
//        }
//        return try session.generateResponse()
//    }
//
//    @available(iOS 13.0.0, *)
//    func generateResponseAsync(prompt: String? = nil) throws -> AsyncThrowingStream<String, any Error> {
//        if let prompt = prompt {
//            try session.addQueryChunk(inputText: prompt)
//        }
//        return session.generateResponseAsync()
//    }
//}



import Foundation

// Mocked replacement for MediaPipeTasksGenAI and MediaPipeTasksGenAIC
final class LlmInference {
    final class Options {
        var modelPath: String
        var maxTokens: Int = 0
        var waitForWeightUploads: Bool = false
        var supportedLoraRanks: [Int]? = nil

        init(modelPath: String) {
            self.modelPath = modelPath
        }
    }

    init(options: Options) throws {
        // Simulate loading model
        print("Mock LlmInference initialized with modelPath: \(options.modelPath)")
    }

    final class Session {
        final class Options {
            var temperature: Float = 0.0
            var randomSeed: Int = 0
            var topk: Int = 0
            var topp: Float? = nil
            var loraPath: String? = nil
        }

        init(llmInference: LlmInference, options: Options) throws {
            // Simulate session setup
            print("Mock Session initialized")
        }

        func sizeInTokens(text: String) throws -> Int {
            // Return word count as token count for mock
            return text.split(separator: " ").count
        }

        func addQueryChunk(inputText: String) throws {
            // Simulate adding input text
            print("Mock addQueryChunk: \(inputText)")
        }

        func generateResponse() throws -> String {
            // Return dummy string
            return "Mocked response"
        }

        @available(iOS 13.0.0, *)
        func generateResponseAsync() -> AsyncThrowingStream<String, any Error> {
            return AsyncThrowingStream { continuation in
                Task {
                    for word in ["Mock", "response", "stream"] {
                        try await Task.sleep(nanoseconds: 200_000_000)
                        continuation.yield(word)
                    }
                    continuation.finish()
                }
            }
        }
    }
}

// ✅ Your existing code remains unchanged — only the import dependency is replaced

struct InferenceModel {
    private(set) var inference: LlmInference

    init(modelPath: String, maxTokens: Int, supportedLoraRanks: [Int]?) throws {
        let fileManager = FileManager.default

        guard let documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw NSError(domain: "InferenceModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Document directory not found"])
        }

        let fileName = (modelPath as NSString).lastPathComponent

        let resolvedPath = documentDirectory.appendingPathComponent(fileName).path

        let llmOptions = LlmInference.Options(modelPath: resolvedPath)
        llmOptions.maxTokens = maxTokens
        llmOptions.waitForWeightUploads = true
        if let supportedLoraRanks = supportedLoraRanks {
            llmOptions.supportedLoraRanks = supportedLoraRanks
        }
        self.inference = try LlmInference(options: llmOptions)
    }
}

final class InferenceSession {
    private let session: LlmInference.Session

    init(inference: LlmInference, temperature: Float, randomSeed: Int, topK: Int, topP: Double? = nil, loraPath: String? = nil) throws {
        let options = LlmInference.Session.Options()
        options.temperature = temperature
        options.randomSeed = randomSeed
        options.topk = topK
        if let topP = topP {
            options.topp = Float(topP)
        }
        if let loraPath = loraPath {
            options.loraPath = loraPath
        }
        self.session = try LlmInference.Session(llmInference: inference, options: options)
    }

    func sizeInTokens(prompt: String) throws -> Int {
        return try session.sizeInTokens(text: prompt)
    }

    func addQueryChunk(prompt: String) throws {
        try session.addQueryChunk(inputText: prompt)
    }

    func generateResponse(prompt: String? = nil) throws -> String {
        if let prompt = prompt {
            try session.addQueryChunk(inputText: prompt)
        }
        return try session.generateResponse()
    }

    @available(iOS 13.0.0, *)
    func generateResponseAsync(prompt: String? = nil) throws -> AsyncThrowingStream<String, any Error> {
        if let prompt = prompt {
            try session.addQueryChunk(inputText: prompt)
        }
        return session.generateResponseAsync()
    }
}
