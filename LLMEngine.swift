// LLMEngine.swift
//
// High-level orchestration layer for on-device LLM inference.
// Owns the LlamaContext lifecycle, serialises concurrent generation requests,
// assembles Gemma chat-template prompts, and provides an on-device ASR path
// with SpeechFallback as a backup.

import Foundation
import Observation
import SwiftUI

/// A single conversation turn with optional image data for multi-turn visual memory.
public struct ConversationTurn: Sendable {
    public let isUser: Bool
    public let content: String
    public let imageData: Data?
    public init(isUser: Bool, content: String, imageData: Data? = nil) {
        self.isUser = isUser
        self.content = content
        self.imageData = imageData
    }
}

@MainActor
@Observable
class LLMEngine {
    static let shared = LLMEngine()

    var llamaContext: LlamaContext?
    var isModelLoaded: Bool = false
    var loadingError: String? = nil
    var isGenerating: Bool = false

    /// True once a multimodal projector has been loaded alongside the text model.
    var supportsVision: Bool { llamaContext?.mtmd != nil }
    var supportsAudio: Bool { llamaContext?.mtmd?.supportsAudio ?? false }

    private var lastSessionTag: String? = nil

    // Serial request queue. While isGenerating is true, new calls to
    // generateStreamResponse are enqueued and drained automatically
    // when the current generation finishes.
    private var pendingQueue: [() -> Void] = []

    private init() {}

    private func drainQueue() {
        guard !isGenerating, !pendingQueue.isEmpty else { return }
        pendingQueue.removeFirst()()
    }

    // MARK: - Cancel

    /// Hard-stop the current generation. The decode loop exits on the next yield.
    func stopGeneration() {
        guard isGenerating else { return }
        llamaContext?.cancelGeneration()
    }

    // MARK: - ASR

    /// Transcribe PCM audio using the model's audio tower with near-greedy sampling.
    /// Falls back to SpeechFallback if output fails a plausibility check.
    func transcribeAudio(
        pcm: [Float],
        language: String,
        sessionTag: String? = nil
    ) async -> String? {
        guard supportsAudio, !pcm.isEmpty else { return nil }
        let canonicalLanguage = Self.normalizeLanguage(language)
        let asrPrompt = """
        Transcribe the following speech segment in \(canonicalLanguage) into \(canonicalLanguage) text.

        Follow these specific instructions for formatting the answer:
        * Only output the transcription, with no newlines.
        * When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
        """

        llamaContext?.samplingOverride = LlamaContext.SamplingOverride(temperature: 0.3, topK: 16, topP: 0.7)

        let modelResult: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            generateStreamResponse(
                systemPrompt: "", history: [], newMessage: asrPrompt,
                audioAttachments: [pcm], enableThinking: false,
                sessionTag: sessionTag.map { "\($0)/asr" } ?? "asr",
                onPartialResult: { _ in },
                onCompletion: { final, _ in
                    let text = (final ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
                    cont.resume(returning: text.isEmpty ? nil : text)
                }
            )
        }
        llamaContext?.samplingOverride = nil

        if let result = modelResult, Self.plausibleTranscription(result, prompt: asrPrompt) {
            return result
        }
        guard let wav = try? PCMAudio.encodeToWAV(pcm: pcm) else { return modelResult }
        return await SpeechFallback.transcribe(wavData: wav, preferredLanguage: language) ?? modelResult
    }

    private static func plausibleTranscription(_ text: String, prompt: String) -> Bool {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard t.count >= 2 else { return false }
        let nonPunct = t.unicodeScalars.filter {
            !CharacterSet.punctuationCharacters.contains($0) &&
            !CharacterSet.whitespacesAndNewlines.contains($0)
        }
        guard !nonPunct.isEmpty else { return false }
        if t.lowercased().contains(String(prompt.prefix(30)).lowercased()) { return false }
        return !["i cannot", "i'm unable", "sorry, i can"].contains(where: { t.lowercased().contains($0) })
    }

    /// Normalise a locale string to the canonical English language name
    /// expected by the Gemma prompt template.
    static func normalizeLanguage(_ raw: String) -> String {
        let s = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty else { return "English" }
        let map: [(keys: [String], canonical: String)] = [
            (["chinese", "mandarin", "zh-cn", "zh-hans", "simplified"], "Chinese"),
            (["cantonese", "zh-hk", "zh-tw", "zh-hant", "traditional"], "Cantonese"),
            (["english", "en-us", "en-gb"], "English"),
            (["japanese", "ja-jp"], "Japanese"),
            (["korean", "ko-kr"], "Korean"),
            (["french", "français", "fr-fr"], "French"),
            (["german", "deutsch", "de-de"], "German"),
            (["spanish", "español", "es-es"], "Spanish"),
            (["russian", "ru-ru"], "Russian"),
            (["italian", "italiano"], "Italian"),
            (["portuguese", "português"], "Portuguese"),
            (["arabic"], "Arabic"),
            (["hindi"], "Hindi")
        ]
        return map.first(where: { $0.keys.contains(where: { s.contains($0) }) })?.canonical ?? "English"
    }

    // MARK: - Model loading

    func loadModelIfNeeded() {
        guard !isModelLoaded, !isGenerating else { return }
        Task.detached(priority: .userInitiated) {
            do {
                let modelPath = GGUFDownloadManager.modelFileURL.path
                guard FileManager.default.fileExists(atPath: modelPath) else {
                    await MainActor.run { self.loadingError = "Model file not found" }
                    return
                }
                // Model loading and GPU compilation are pure C calls — safe on a detached task.
                let context = try LlamaContext.create_context(path: modelPath)
                let mmprojPath = GGUFDownloadManager.mmprojFileURL.path
                if FileManager.default.fileExists(atPath: mmprojPath) {
                    if !context.loadMMProj(path: mmprojPath) {
                        print("mmproj load failed — image/audio input will be unavailable")
                    }
                }
                await MainActor.run { self.llamaContext = context; self.isModelLoaded = true }
            } catch {
                await MainActor.run { self.loadingError = "Load failed: \(error.localizedDescription)" }
            }
        }
    }

    // MARK: - Generation

    /// Stream a response from the model.
    ///
    /// - Parameters:
    ///   - systemPrompt: Injected as the first turn if non-empty.
    ///   - history: Prior conversation turns (up to 100 are used).
    ///   - newMessage: The current user message.
    ///   - imageAttachments: JPEG/PNG/HEIC image bytes for the current turn.
    ///   - audioAttachments: PCM F32 audio arrays for the current turn.
    ///   - enableThinking: Prepend `<|think|>` to activate reasoning mode.
    ///   - sessionTag: Optional label for session-isolation logging only.
    ///   - onPartialResult: Called on the main thread with each answer token.
    ///   - onThinkingChunk: Called on the main thread with reasoning tokens (thinking mode only).
    ///   - onCompletion: Called on the main thread when generation finishes.
    func generateStreamResponse(
        systemPrompt: String,
        history: [ConversationTurn],
        newMessage: String,
        imageAttachments: [Data] = [],
        audioAttachments: [[Float]] = [],
        enableThinking: Bool = false,
        sessionTag: String? = nil,
        onPartialResult: @escaping @MainActor @Sendable (String) -> Void,
        onThinkingChunk: (@MainActor @Sendable (String) -> Void)? = nil,
        onCompletion: @escaping @MainActor @Sendable (String?, Error?) -> Void
    ) {
        guard let context = llamaContext else {
            onCompletion(nil, NSError(domain: "LLMEngine", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model not loaded"]))
            return
        }
        if isGenerating {
            pendingQueue.append({ [weak self] in
                self?.generateStreamResponse(
                    systemPrompt: systemPrompt, history: history, newMessage: newMessage,
                    imageAttachments: imageAttachments, audioAttachments: audioAttachments,
                    enableThinking: enableThinking, sessionTag: sessionTag,
                    onPartialResult: onPartialResult, onThinkingChunk: onThinkingChunk,
                    onCompletion: onCompletion)
            })
            return
        }
        isGenerating = true
        if let tag = sessionTag, tag != lastSessionTag {
            print("Session switch: \(lastSessionTag ?? "nil") -> \(tag) (KV cache reset)")
            lastSessionTag = tag
        }

        Task.detached(priority: .userInitiated) {
            let effectiveSystem = enableThinking ? "<|think|>\n\(systemPrompt)" : systemPrompt
            var prompt = effectiveSystem.isEmpty ? "" : "<|turn>system\n\(effectiveSystem)<turn|>\n"

            let historyClipped = Array(history.suffix(100))
            let (mtmdOn, audioOn) = await MainActor.run { (context.mtmd != nil, context.mtmd?.supportsAudio ?? false) }

            // Collect up to 10 historical images for re-injection.
            var histImgEntries: [(idx: Int, data: Data)] = []
            if mtmdOn {
                for (i, h) in historyClipped.enumerated() {
                    if h.isUser, let img = h.imageData, !img.isEmpty { histImgEntries.append((i, img)) }
                }
                histImgEntries = Array(histImgEntries.suffix(10))
            }
            let keptIdx = Set(histImgEntries.map { $0.idx })
            let histBitmaps = histImgEntries.map { $0.data }
            let marker = mtmdOn ? await MainActor.run { MTMDContext.mediaMarker } : ""

            for (i, msg) in historyClipped.enumerated() {
                var content = msg.content
                if msg.isUser, keptIdx.contains(i), !marker.isEmpty { content = "\(marker)\n" + content }
                prompt += "<|turn>\(msg.isUser ? "user" : "model")\n\(content)<turn|>\n"
            }

            let useImg = !imageAttachments.isEmpty && mtmdOn
            let useAud = !audioAttachments.isEmpty && mtmdOn && audioOn
            let useMTMD = !histBitmaps.isEmpty || useImg || useAud
            var mediaPrefix = ""
            if useMTMD && !marker.isEmpty {
                if useImg { mediaPrefix += String(repeating: "\(marker)\n", count: imageAttachments.count) }
                if useAud { mediaPrefix += String(repeating: "\(marker)\n", count: audioAttachments.count) }
            }
            prompt += "<|turn>user\n\(mediaPrefix)\(newMessage)<turn|>\n<|turn>model\n"

            let allImages = histBitmaps + imageAttachments
            await MainActor.run { context.reset() }

            do {
                let state = GenerationState()
                let stopWords = ["<turn|>", "<|turn>", "model\n", "</s>"]
                let thinkOpen = "<|channel>thought\n"
                let thinkClose = "<channel|>"

                enum Mode { case initial, thinking, answering }
                final class ModeBox: @unchecked Sendable { var v: Mode; init(_ m: Mode) { v = m } }
                let mode = ModeBox(enableThinking ? .initial : .answering)

                let handler: @Sendable (String) -> Bool = { chunk in
                    if context.cancelFlag.isCancelled { return false }
                    state.appendToBuffer(chunk)

                    if mode.v == .initial {
                        let buf = state.textBuffer
                        if let r = buf.range(of: thinkOpen) {
                            _ = state.extractSafeText(upTo: r.upperBound); mode.v = .thinking
                        } else if buf.count > 64 { mode.v = .answering } else { return true }
                    }

                    if mode.v == .thinking {
                        let buf = state.textBuffer
                        if let r = buf.range(of: thinkClose) {
                            let tail = String(buf[..<r.lowerBound])
                            if !tail.isEmpty { Task { @MainActor in onThinkingChunk?(tail) } }
                            _ = state.extractSafeText(upTo: r.upperBound); mode.v = .answering
                        } else {
                            if buf.count > thinkClose.count {
                                let si = buf.index(buf.startIndex, offsetBy: buf.count - thinkClose.count)
                                let c = state.extractSafeText(upTo: si)
                                if !c.isEmpty { Task { @MainActor in onThinkingChunk?(c) } }
                            }
                            return true
                        }
                    }

                    let buf = state.textBuffer
                    for sw in stopWords {
                        if let r = buf.range(of: sw) {
                            let safe = String(buf[..<r.lowerBound])
                            if !safe.isEmpty { Task { @MainActor in onPartialResult(safe) }; state.appendToFullResponse(safe) }
                            return false
                        }
                    }
                    if buf.count > 3 {
                        let si = buf.index(buf.startIndex, offsetBy: buf.count - 3)
                        let c = state.extractSafeText(upTo: si)
                        Task { @MainActor in onPartialResult(c) }; state.appendToFullResponse(c)
                    }
                    return true
                }

                if !allImages.isEmpty && useAud {
                    try await context.generateIncrementalResponseWithMixedMedia(newText: prompt, images: allImages, audioSamples: audioAttachments, onToken: handler)
                } else if !allImages.isEmpty {
                    try await context.generateIncrementalResponseWithImages(newText: prompt, images: allImages, onToken: handler)
                } else if useAud {
                    try await context.generateIncrementalResponseWithAudio(newText: prompt, audioSamples: audioAttachments, onToken: handler)
                } else {
                    try await context.generateIncrementalResponse(newText: prompt, onToken: handler)
                }

                let tail = state.textBuffer
                if !tail.isEmpty {
                    if mode.v == .thinking { Task { @MainActor in onThinkingChunk?(tail) } }
                    else { Task { @MainActor in onPartialResult(tail) }; state.appendToFullResponse(tail) }
                }
                let final = state.fullResponse
                // Clear isGenerating before calling onCompletion so any queued request
                // triggered inside onCompletion is not dropped.
                await MainActor.run { self.isGenerating = false; onCompletion(final, nil); self.drainQueue() }
            } catch {
                await MainActor.run { self.isGenerating = false; onCompletion(nil, error); self.drainQueue() }
            }
        }
    }
}
