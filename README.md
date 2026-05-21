# iOS-LLMKit

A Swift orchestration layer for on-device LLM inference via [LlamaSwift](https://github.com/EinarInCanada/LlamaSwift). Manages the model lifecycle, serialises concurrent generation requests, assembles chat-template prompts, and provides an on-device ASR path with a system fallback.

## Features

- **LLMEngine** — `@MainActor @Observable` engine with serial request queue; new requests are enqueued automatically while a generation is in progress
- **Streaming generation** — token-by-token callbacks with optional deep-thinking mode (routes thinking tokens separately from answer tokens)
- **Multimodal** — passes image and audio attachments through libmtmd when the loaded model supports them; re-injects up to 10 historical images for multi-turn visual memory
- **On-device ASR** — transcribes PCM audio via the model's audio tower with near-greedy sampling; falls back to Apple `SFSpeechRecognizer` on plausibility failure
- **PCMAudio** — waveform visualisation, WAV encoding, resampling (via `AVAudioConverter`), silence trimming, and peak normalisation
- **GenerationState** — thread-safe token buffer guarded by `NSLock` for use across `Task.detached` and main-actor boundaries

## Requirements

- iOS 17+
- Swift 5.9+
- [LlamaSwift](https://github.com/EinarInCanada/LlamaSwift) (provides `LlamaContext` and `MTMDContext`)
- [GGUFDownloader](https://github.com/EinarInCanada/GGUFDownloader) (provides `GGUFDownloadManager`)

## Installation

Add LLMKit as a local package alongside its dependencies. In your app target:

```swift
// Package.swift or Xcode file browser
.package(path: "../LLMKit")
.package(path: "../LlamaSwift")
.package(path: "../GGUFDownloader")
```

## Usage

### Load the model

```swift
LLMEngine.shared.loadModelIfNeeded()
// Observe LLMEngine.shared.isModelLoaded / loadingError in SwiftUI
```

### Stream a text response

```swift
LLMEngine.shared.generateStreamResponse(
    systemPrompt: "You are a helpful assistant.",
    history: conversationHistory,
    newMessage: userInput,
    onPartialResult: { token in outputText += token },
    onCompletion: { final, error in /* done */ }
)
```

### Stream with images

```swift
LLMEngine.shared.generateStreamResponse(
    systemPrompt: "",
    history: [],
    newMessage: "Describe this image.",
    imageAttachments: [jpegData],
    onPartialResult: { token in outputText += token },
    onCompletion: { final, error in }
)
```

### Thinking mode

```swift
LLMEngine.shared.generateStreamResponse(
    systemPrompt: systemPrompt,
    history: history,
    newMessage: message,
    enableThinking: true,
    onPartialResult: { token in answerText += token },
    onThinkingChunk: { chunk in thinkingText += chunk },
    onCompletion: { final, error in }
)
```

### On-device ASR

```swift
// pcm: [Float] at 16 kHz mono
let transcript = await LLMEngine.shared.transcribeAudio(pcm: pcmSamples, language: "English")
```

Falls back to `SFSpeechRecognizer` if the model output fails a plausibility check.

### Stop generation

```swift
LLMEngine.shared.stopGeneration()
```

## Architecture

```
LLMEngine (singleton, @MainActor @Observable)
├── LlamaContext          — llama.cpp inference (from LlamaSwift)
├── MTMDContext           — multimodal embedding (from LlamaSwift)
├── GGUFDownloadManager   — model file location (from GGUFDownloader)
├── GenerationState       — thread-safe token buffer
├── PCMAudio              — audio utilities
└── SpeechFallback        — Apple SFSpeechRecognizer wrapper
```

### Prompt format

LLMKit uses the Gemma 4 chat template:

```
<|turn>system
{system}<turn|>
<|turn>user
{user message}<turn|>
<|turn>model
```

Media markers (e.g. `<image>`) are inserted before user content when multimodal attachments are present.

### Serial queue

`generateStreamResponse` sets `isGenerating = true` when a generation starts. Calls made while `isGenerating` is true are appended to `pendingQueue` and drained automatically after each generation completes — callers do not need to poll or coordinate.

### Thinking mode state machine

With `enableThinking: true`, the engine prepends `<|think|>` and routes tokens through a three-state machine:

1. `.initial` — scan for `<|channel>thought\n` opener
2. `.thinking` — emit to `onThinkingChunk` until `<channel|>` closes the block
3. `.answering` — emit to `onPartialResult` as normal

## PCMAudio utilities

| Method | Purpose |
|---|---|
| `computeBars(pcm:binCount:)` | Downsample to N peak bars for waveform UI |
| `encodeBars(_:)` / `decodeBars(_:)` | Serialise bars as raw Float32 bytes |
| `encodeToWAV(pcm:sampleRate:)` | Lossless 16-bit PCM WAV for SFSpeechRecognizer |
| `resample(_:from:to:)` | Resample between arbitrary rates via AVAudioConverter |
| `trimSilence(_:threshold:windowMs:sampleRate:)` | Remove leading/trailing silence |
| `normalizePCM(_:targetPeak:)` | Peak-normalise quiet recordings (gain capped at 8x) |

## License

MIT — see [LICENSE](LICENSE).
