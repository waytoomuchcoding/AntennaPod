package de.danoeh.antennapod.net.download.service.episode;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class AacClipExtractor implements AutoCloseable {
    private final MediaExtractor extractor = new MediaExtractor();
    private final MediaFormat format;
    private final long durationMs;

    interface SampleReader {
        void seekTo(long timeUs);

        long getSampleTime();

        int readSampleData(ByteBuffer buffer);

        void advance();
    }

    interface SampleWriter {
        void writeSample(ByteBuffer buffer, int size, long presentationTimeUs);
    }

    private AacClipExtractor(@NonNull File file) throws IOException {
        extractor.setDataSource(file.getAbsolutePath());
        MediaFormat audioFormat = null;
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat trackFormat = extractor.getTrackFormat(i);
            if (MediaFormat.MIMETYPE_AUDIO_AAC.equals(trackFormat.getString(MediaFormat.KEY_MIME))) {
                extractor.selectTrack(i);
                audioFormat = trackFormat;
                break;
            }
        }
        format = audioFormat;
        durationMs = audioFormat != null && audioFormat.containsKey(MediaFormat.KEY_DURATION)
                ? audioFormat.getLong(MediaFormat.KEY_DURATION) / 1000 : 0;
    }

    /**
     * Opens the file if it contains an AAC audio track, otherwise returns null.
     */
    @Nullable
    public static AacClipExtractor open(@NonNull File file) {
        try {
            AacClipExtractor clipExtractor = new AacClipExtractor(file);
            if (clipExtractor.format != null) {
                return clipExtractor;
            }
            clipExtractor.close();
        } catch (IOException | RuntimeException e) {
            return null;
        }
        return null;
    }

    public long getDurationMs() {
        return durationMs;
    }

    /**
     * Writes the audio between startMs and endMs to an M4A file.
     *
     * @return the timestamp in milliseconds where the written clip actually starts
     */
    public long writeClip(long startMs, long endMs, @NonNull File output) throws IOException {
        MediaMuxer muxer = new MediaMuxer(output.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        try {
            int track = muxer.addTrack(format);
            muxer.start();
            int bufferSize = format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)
                    ? format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE) : 64 * 1024;
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            long firstSampleUs = copySamples(new SampleReader() {
                @Override
                public void seekTo(long timeUs) {
                    extractor.seekTo(timeUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
                }

                @Override
                public long getSampleTime() {
                    return extractor.getSampleTime();
                }

                @Override
                public int readSampleData(ByteBuffer buffer) {
                    return extractor.readSampleData(buffer, 0);
                }

                @Override
                public void advance() {
                    extractor.advance();
                }
            }, (buffer, size, presentationTimeUs) -> {
                info.set(0, size, presentationTimeUs, MediaCodec.BUFFER_FLAG_KEY_FRAME);
                muxer.writeSampleData(track, buffer, info);
            }, startMs * 1000, endMs * 1000, bufferSize);
            muxer.stop();
            return Math.max(0, firstSampleUs) / 1000;
        } finally {
            muxer.release();
        }
    }

    /**
     * Copies the samples between startUs and endUs, shifting their timestamps so that the first one starts at 0.
     *
     * @return the original timestamp of the first copied sample, or -1 if no sample was copied
     */
    static long copySamples(SampleReader reader, SampleWriter writer, long startUs, long endUs, int bufferSize) {
        reader.seekTo(startUs);
        ByteBuffer buffer = ByteBuffer.allocate(bufferSize);
        long firstSampleUs = -1;
        while (true) {
            long sampleTimeUs = reader.getSampleTime();
            if (sampleTimeUs < 0 || sampleTimeUs >= endUs) {
                break;
            }
            buffer.clear();
            int size = reader.readSampleData(buffer);
            if (size < 0) {
                break;
            }
            if (firstSampleUs < 0) {
                firstSampleUs = sampleTimeUs;
            }
            writer.writeSample(buffer, size, sampleTimeUs - firstSampleUs);
            reader.advance();
        }
        return firstSampleUs;
    }

    @Override
    public void close() {
        extractor.release();
    }
}
