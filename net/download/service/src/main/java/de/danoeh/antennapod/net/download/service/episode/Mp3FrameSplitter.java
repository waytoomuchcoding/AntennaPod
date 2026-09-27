package de.danoeh.antennapod.net.download.service.episode;

import androidx.annotation.NonNull;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class Mp3FrameSplitter {
    private static final int[][] BITRATES = {
            {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320},
            {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160}};
    private static final int[][] SAMPLE_RATES = {
            {11025, 12000, 8000}, {0, 0, 0}, {22050, 24000, 16000}, {44100, 48000, 32000}};
    private static final int MIN_FRAMES = 100;

    public static class Chunk {
        public final long byteStart;
        public final long byteEnd;
        public final long startMs;
        public final long endMs;

        Chunk(long byteStart, long byteEnd, long startMs, long endMs) {
            this.byteStart = byteStart;
            this.byteEnd = byteEnd;
            this.startMs = startMs;
            this.endMs = endMs;
        }
    }

    private Mp3FrameSplitter() {
    }

    /**
     * Splits an MP3 file into chunks of roughly the given duration at frame boundaries.
     * Returns an empty list if the file does not look like an MP3 file.
     */
    @NonNull
    public static List<Chunk> split(@NonNull File file, long chunkMs) throws IOException {
        List<Chunk> chunks = new ArrayList<>();
        try (InputStream in = new BufferedInputStream(new FileInputStream(file), 65536)) {
            long pos = skipId3(in);
            byte[] header = new byte[4];
            long chunkByteStart = -1;
            double chunkStartMs = 0;
            double timeMs = 0;
            int frames = 0;
            while (true) {
                in.mark(4);
                if (readFully(in, header) < 4) {
                    break;
                }
                int frameLength = getFrameLength(header);
                if (frameLength <= 0) {
                    in.reset();
                    if (in.skip(1) != 1) {
                        break;
                    }
                    pos++;
                    continue;
                }
                if (chunkByteStart < 0) {
                    chunkByteStart = pos;
                }
                frames++;
                timeMs += getFrameDurationMs(header);
                long toSkip = frameLength - 4;
                while (toSkip > 0) {
                    long skipped = in.skip(toSkip);
                    if (skipped <= 0) {
                        break;
                    }
                    toSkip -= skipped;
                }
                pos += frameLength;
                if (timeMs - chunkStartMs >= chunkMs) {
                    chunks.add(new Chunk(chunkByteStart, pos, Math.round(chunkStartMs), Math.round(timeMs)));
                    chunkByteStart = pos;
                    chunkStartMs = timeMs;
                }
            }
            if (chunkByteStart >= 0 && pos > chunkByteStart && timeMs > chunkStartMs) {
                chunks.add(new Chunk(chunkByteStart, Math.min(pos, file.length()),
                        Math.round(chunkStartMs), Math.round(timeMs)));
            }
            if (frames < MIN_FRAMES) {
                chunks.clear();
            }
        }
        return chunks;
    }

    private static long skipId3(InputStream in) throws IOException {
        in.mark(10);
        byte[] id3 = new byte[10];
        if (readFully(in, id3) == 10 && id3[0] == 'I' && id3[1] == 'D' && id3[2] == '3') {
            long size = ((id3[6] & 0x7f) << 21) | ((id3[7] & 0x7f) << 14) | ((id3[8] & 0x7f) << 7) | (id3[9] & 0x7f);
            long toSkip = size;
            while (toSkip > 0) {
                long skipped = in.skip(toSkip);
                if (skipped <= 0) {
                    break;
                }
                toSkip -= skipped;
            }
            return 10 + size;
        }
        in.reset();
        return 0;
    }

    private static int readFully(InputStream in, byte[] buffer) throws IOException {
        int read = 0;
        while (read < buffer.length) {
            int n = in.read(buffer, read, buffer.length - read);
            if (n < 0) {
                break;
            }
            read += n;
        }
        return read;
    }

    static int getFrameLength(byte[] header) {
        if ((header[0] & 0xff) != 0xff || (header[1] & 0xe0) != 0xe0) {
            return -1;
        }
        int version = (header[1] >> 3) & 3;
        int layer = (header[1] >> 1) & 3;
        int bitrateIndex = (header[2] >> 4) & 0xf;
        int sampleRateIndex = (header[2] >> 2) & 3;
        int padding = (header[2] >> 1) & 1;
        if (layer != 1 || version == 1 || bitrateIndex == 0 || bitrateIndex == 15 || sampleRateIndex == 3) {
            return -1;
        }
        int bitrate = BITRATES[version == 3 ? 0 : 1][bitrateIndex] * 1000;
        int sampleRate = SAMPLE_RATES[version][sampleRateIndex];
        return (version == 3 ? 144 : 72) * bitrate / sampleRate + padding;
    }

    private static double getFrameDurationMs(byte[] header) {
        int version = (header[1] >> 3) & 3;
        int sampleRate = SAMPLE_RATES[version][(header[2] >> 2) & 3];
        return (version == 3 ? 1152 : 576) * 1000.0 / sampleRate;
    }
}
