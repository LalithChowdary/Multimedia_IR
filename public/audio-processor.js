// public/audio-processor.js

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // A buffer to store audio data until we have a full chunk to send
        this.buffer = new Int16Array(1024);
        this.bufferIndex = 0;
    }

    /**
     * This method is called by the browser's audio engine with new audio data.
     * @param {Float32Array[][]} inputs - An array of inputs, each with an array of channels.
     *                                    We expect a single input with a single channel.
     */
    process(inputs) {
        // We expect mono audio, so we only take the first channel of the first input.
        const inputChannel = inputs[0][0];

        // If there's no audio data, we don't need to do anything.
        if (!inputChannel) {
            return true;
        }

        // The input is a Float32Array ranging from -1.0 to 1.0.
        // We need to convert it to a 16-bit integer format (Int16Array)
        // ranging from -32768 to 32767, which is what our backend expects.
        for (let i = 0; i < inputChannel.length; i++) {
            // Clamp the value to be within [-1, 1] before conversion
            const s = Math.max(-1, Math.min(1, inputChannel[i]));
            // Convert to 16-bit integer
            this.buffer[this.bufferIndex++] = s < 0 ? s * 0x8000 : s * 0x7FFF;

            // When the buffer is full, send it to the main thread.
            if (this.bufferIndex === this.buffer.length) {
                this.port.postMessage(this.buffer.buffer);
                // Reset the buffer index for the next chunk.
                this.bufferIndex = 0;
            }
        }

        // Return true to keep the processor alive.
        return true;
    }
}

// Register the processor with the name 'audio-processor'.
// This name is used in the main thread to load the worklet.
registerProcessor('audio-processor', AudioProcessor);
