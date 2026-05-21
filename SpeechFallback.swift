// SpeechFallback.swift
//
// On-device fallback transcription using Apple's SFSpeechRecognizer.
// Used when the primary on-device ASR path is unavailable or produces
// implausible output. Prefers on-device recognition to avoid sending
// audio to Apple servers.

import Foundation
import Speech

private final class TranscriptionGate: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func tryFire(_ work: () -> Void) {
        lock.lock(); defer { lock.unlock() }
        guard !done else { return }
        done = true; work()
    }
}

enum SpeechFallback {

    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
    }

    /// Transcribe WAV audio using iOS SFSpeechRecognizer.
    /// Returns nil if authorization is denied, the recognizer is unavailable,
    /// or recognition fails.
    static func transcribe(wavData: Data, preferredLanguage: String? = nil) async -> String? {
        guard await requestAuthorization() else { return nil }
        let locale = pickLocale(preferred: preferredLanguage)
        guard let recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer(),
              recognizer.isAvailable else { return nil }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("stt-fallback-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        guard (try? wavData.write(to: tempURL, options: .atomic)) != nil else { return nil }

        let request = SFSpeechURLRecognitionRequest(url: tempURL)
        request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        request.shouldReportPartialResults = false

        return await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let gate = TranscriptionGate()
            _ = recognizer.recognitionTask(with: request) { result, error in
                if error != nil { gate.tryFire { cont.resume(returning: nil) }; return }
                guard let result, result.isFinal else { return }
                let text = result.bestTranscription.formattedString
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                gate.tryFire { cont.resume(returning: text.isEmpty ? nil : text) }
            }
        }
    }

    private static func pickLocale(preferred: String?) -> Locale {
        guard let pref = preferred?.lowercased(), !pref.isEmpty else { return .current }
        let map: [(keys: [String], id: String)] = [
            (["chinese", "mandarin", "zh-cn", "zh-hans", "simplified"], "zh-CN"),
            (["cantonese", "zh-hk", "zh-tw", "zh-hant", "traditional"], "zh-HK"),
            (["english", "en-us", "en-gb"], "en-US"),
            (["japanese", "ja-jp"], "ja-JP"),
            (["korean", "ko-kr"], "ko-KR"),
            (["french", "français", "fr-fr"], "fr-FR"),
            (["german", "deutsch", "de-de"], "de-DE"),
            (["spanish", "español", "es-es"], "es-ES"),
            (["russian", "ru-ru"], "ru-RU")
        ]
        for entry in map where entry.keys.contains(where: { pref.contains($0) }) {
            return Locale(identifier: entry.id)
        }
        return .current
    }
}
