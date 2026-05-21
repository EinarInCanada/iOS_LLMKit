import Foundation

// Thread-safe token buffer used by the generation stream handler.
// NSLock guards both _textBuffer and _fullResponse since they are written
// from a detached Task and read from the main thread.
final class GenerationState: @unchecked Sendable {
    private let lock = NSLock()

    // nonisolated(unsafe) tells Swift 6 that thread safety is handled manually
    // via NSLock, so the compiler should not infer main-actor isolation.
    nonisolated(unsafe) private var _textBuffer = ""
    nonisolated(unsafe) private var _fullResponse = ""

    init() {}

    var textBuffer: String { lock.withLock { _textBuffer } }
    var fullResponse: String { lock.withLock { _fullResponse } }

    func appendToBuffer(_ text: String) { lock.withLock { _textBuffer += text } }
    func appendToFullResponse(_ text: String) { lock.withLock { _fullResponse += text } }

    func extractSafeText(upTo index: String.Index) -> String {
        lock.lock()
        defer { lock.unlock() }
        let extracted = String(_textBuffer[..<index])
        _textBuffer = String(_textBuffer[index...])
        return extracted
    }
}
