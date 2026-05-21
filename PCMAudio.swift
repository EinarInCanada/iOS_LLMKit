// PCMAudio.swift
//
// PCM audio utilities for on-device voice input:
//   - computeBars:     downsample 16 kHz mono F32 PCM to a normalised waveform bar array
//   - encodeBars / decodeBars: serialise waveform bars as raw Float32 bytes
//   - encodeToWAV:     lossless 16-bit PCM WAV encoder
//   - resample:        resample PCM between arbitrary sample rates via AVAudioConverter
//   - trimSilence:     remove leading/trailing silence below an amplitude threshold
//   - normalizePCM:    peak-normalise quiet recordings before ASR

import Foundation
import AVFoundation

enum PCMAudio {
    static let defaultSampleRate: Double = 16000
    static let defaultBarCount: Int = 64

    // MARK: - Waveform sampling

    /// Downsample PCM to `binCount` peak values in [0, 1].
    /// Peak (rather than RMS) is used because speech visualisation cares more
    /// about voiced/unvoiced transitions than average energy.
    static func computeBars(pcm: [Float], binCount: Int = defaultBarCount) -> [Float] {
        guard !pcm.isEmpty, binCount > 0 else { return [] }
        let binSize = max(1, pcm.count / binCount)
        var bars: [Float] = []
        bars.reserveCapacity(binCount)
        var i = 0
        while i < pcm.count && bars.count < binCount {
            let end = min(i + binSize, pcm.count)
            var peak: Float = 0
            for j in i..<end { let v = abs(pcm[j]); if v > peak { peak = v } }
            bars.append(peak)
            i = end
        }
        let maxV = bars.max() ?? 1
        if maxV > 0.001 { for k in 0..<bars.count { bars[k] /= maxV } }
        return bars
    }

    // MARK: - Waveform serialisation

    static func encodeBars(_ bars: [Float]) -> Data {
        bars.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    static func decodeBars(_ data: Data?) -> [Float] {
        guard let data, !data.isEmpty else { return [] }
        let count = data.count / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        var bars = [Float](repeating: 0, count: count)
        bars.withUnsafeMutableBytes { _ = data.copyBytes(to: $0) }
        return bars
    }

    // MARK: - WAV encoding

    /// Encode 16 kHz mono F32 PCM as lossless 16-bit PCM WAV.
    /// WAV preserves the full frequency range needed by audio-tower models —
    /// lossy codecs discard high-frequency content (sibilants, plosives) that
    /// the model relies on for accurate transcription.
    static func encodeToWAV(pcm: [Float], sampleRate: Double = defaultSampleRate) throws -> Data {
        guard !pcm.isEmpty else {
            throw NSError(domain: "PCMAudio", code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Empty PCM input"])
        }
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("pcm-audio-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        let file = try AVAudioFile(forWriting: tempURL, settings: settings)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                            frameCapacity: AVAudioFrameCount(pcm.count)) else {
            throw NSError(domain: "PCMAudio", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not allocate PCM buffer"])
        }
        buffer.frameLength = AVAudioFrameCount(pcm.count)
        if let dst = buffer.floatChannelData?[0] {
            pcm.withUnsafeBufferPointer { if let base = $0.baseAddress { memcpy(dst, base, pcm.count * MemoryLayout<Float>.size) } }
        }
        try file.write(from: buffer)
        return try Data(contentsOf: tempURL)
    }

    // MARK: - Duration

    static func duration(pcmCount: Int, sampleRate: Double = defaultSampleRate) -> Double {
        guard sampleRate > 0 else { return 0 }
        return Double(pcmCount) / sampleRate
    }

    // MARK: - Resampling

    /// Resample mono PCM between arbitrary sample rates using AVAudioConverter.
    /// Typical use: downsample hardware 48 kHz capture to the 16 kHz expected
    /// by on-device audio-tower models and iOS Speech.
    static func resample(_ pcm: [Float], from src: Double, to dst: Double) -> [Float] {
        guard !pcm.isEmpty, src > 0, dst > 0, abs(src - dst) >= 1 else { return pcm }
        guard let srcFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: src, channels: 1, interleaved: false),
              let dstFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: dst, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: srcFmt, to: dstFmt),
              let srcBuf = AVAudioPCMBuffer(pcmFormat: srcFmt, frameCapacity: AVAudioFrameCount(pcm.count)) else {
            return pcm
        }
        srcBuf.frameLength = AVAudioFrameCount(pcm.count)
        if let dstPtr = srcBuf.floatChannelData?[0] {
            pcm.withUnsafeBufferPointer { if let base = $0.baseAddress { memcpy(dstPtr, base, pcm.count * MemoryLayout<Float>.size) } }
        }
        let outCapacity = AVAudioFrameCount(Double(pcm.count) * dst / src) + 64
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: outCapacity) else { return pcm }
        var provided = false
        let status = converter.convert(to: outBuf, error: nil) { _, outStatus in
            if provided { outStatus.pointee = .endOfStream; return nil }
            provided = true; outStatus.pointee = .haveData; return srcBuf
        }
        guard status != .error, let ch = outBuf.floatChannelData?[0], outBuf.frameLength > 0 else { return pcm }
        return Array(UnsafeBufferPointer(start: ch, count: Int(outBuf.frameLength)))
    }

    // MARK: - Silence trimming

    /// Remove leading and trailing silence below `threshold` amplitude.
    /// An 80 ms boundary buffer is preserved to avoid clipping the first and
    /// last phonemes.
    static func trimSilence(
        _ pcm: [Float],
        threshold: Float = 0.015,
        windowMs: Int = 50,
        sampleRate: Double = defaultSampleRate
    ) -> [Float] {
        guard !pcm.isEmpty else { return pcm }
        let windowSize = max(1, Int(Double(windowMs) * sampleRate / 1000))
        var head = 0
        while head + windowSize <= pcm.count {
            if pcm[head..<(head + windowSize)].contains(where: { abs($0) >= threshold }) { break }
            head += windowSize
        }
        var tail = pcm.count
        while tail - windowSize > head {
            if pcm[(tail - windowSize)..<tail].contains(where: { abs($0) >= threshold }) { break }
            tail -= windowSize
        }
        guard head < tail else { return pcm }
        let pad = max(1, Int(0.08 * sampleRate))
        return Array(pcm[max(0, head - pad)..<min(pcm.count, tail + pad)])
    }

    // MARK: - Peak normalisation

    /// Peak-normalise quiet PCM toward `targetPeak`. Gain is capped at 8x.
    /// Recordings already at or above `targetPeak` are left unchanged.
    static func normalizePCM(_ pcm: [Float], targetPeak: Float = 0.7) -> [Float] {
        guard !pcm.isEmpty else { return pcm }
        let peak = pcm.map { abs($0) }.max() ?? 0
        guard peak > 0.001 else { return pcm }
        let gain = targetPeak / peak
        guard gain > 1.05 else { return pcm }
        return pcm.map { $0 * min(gain, 8.0) }
    }
}
