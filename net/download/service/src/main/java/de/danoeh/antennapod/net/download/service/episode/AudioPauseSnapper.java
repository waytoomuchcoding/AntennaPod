package de.danoeh.antennapod.net.download.service.episode;

import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.os.Build;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;

final class AudioPauseSnapper {
    static final long FRAME_STEP_MS = 50L;
    private static final long ENDPOINT_MARGIN_MS = 2000L;
    private static final long OUTER_WINDOW_MS = 1500L;
    private static final long INNER_WINDOW_MS = 3500L;
    private static final int PAUSE_FRAMES = 5;
    private static final double MIN_DROP_DB = 10.0;
    private static final double DISTANCE_PENALTY_DB = 3.5;
    private static final long CODEC_TIMEOUT_US = 10_000L;
    private static final int MAX_CODEC_STEPS = 4000;

    private static final class PcmConfig {
        int sampleRate;
        int channels;
        int encoding = AudioFormat.ENCODING_PCM_16BIT;

        PcmConfig(@NonNull MediaFormat format) {
            sampleRate = format.containsKey(MediaFormat.KEY_SAMPLE_RATE)
                    ? Math.max(1, format.getInteger(MediaFormat.KEY_SAMPLE_RATE)) : 44100;
            channels = format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                    ? Math.max(1, format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)) : 1;
        }

        void update(@NonNull MediaFormat outFormat) {
            if (outFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                sampleRate = Math.max(1, outFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE));
            }
            if (outFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                channels = Math.max(1, outFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT));
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N
                    && outFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                encoding = outFormat.getInteger(MediaFormat.KEY_PCM_ENCODING);
            }
        }
    }

    private AudioPauseSnapper() {
    }

    static long refineBoundary(@NonNull File file, long candidateMs, boolean isStart,
                               long durationMs, long fallbackCushionMs) {
        if (candidateMs <= ENDPOINT_MARGIN_MS
                || (durationMs > 0 && candidateMs >= durationMs - ENDPOINT_MARGIN_MS)) {
            return candidateMs;
        }
        long winStartMs = Math.max(0, candidateMs - (isStart ? OUTER_WINDOW_MS : INNER_WINDOW_MS));
        long rawEndMs = candidateMs + (isStart ? INNER_WINDOW_MS : OUTER_WINDOW_MS);
        long winEndMs = durationMs > 0 ? Math.min(durationMs, rawEndMs) : rawEndMs;
        if (winEndMs - winStartMs < FRAME_STEP_MS * (PAUSE_FRAMES + 2)) {
            return applyFallbackCushion(candidateMs, isStart, durationMs, fallbackCushionMs);
        }
        float[] rms = extractRmsEnvelope(file, winStartMs, winEndMs);
        if (rms == null || rms.length < PAUSE_FRAMES + 2) {
            return applyFallbackCushion(candidateMs, isStart, durationMs, fallbackCushionMs);
        }
        long snapped = snapToPause(candidateMs, isStart, winStartMs, rms, FRAME_STEP_MS, fallbackCushionMs);
        return durationMs > 0 ? Math.min(durationMs, Math.max(0, snapped)) : Math.max(0, snapped);
    }

    static long snapToPause(long candidateMs, boolean isStart, long windowStartMs,
                            @NonNull float[] frameRms, long frameStepMs, long fallbackCushionMs) {
        if (frameRms.length < PAUSE_FRAMES) {
            return Math.max(0, candidateMs + (isStart ? fallbackCushionMs : -fallbackCushionMs));
        }
        float[] sorted = Arrays.copyOf(frameRms, frameRms.length);
        Arrays.sort(sorted);
        double medianRms = sorted[sorted.length / 2];
        if (medianRms <= 1e-5) {
            return candidateMs;
        }
        double medianDb = 20.0 * Math.log10(medianRms + 1e-12);
        double bestScore = Double.MAX_VALUE;
        long bestMs = -1;
        for (int i = 0; i <= frameRms.length - PAUSE_FRAMES; i++) {
            double sum = 0;
            for (int k = 0; k < PAUSE_FRAMES; k++) {
                sum += frameRms[i + k];
            }
            double pauseDb = 20.0 * Math.log10(sum / PAUSE_FRAMES + 1e-12);
            if (medianDb - pauseDb < MIN_DROP_DB) {
                continue;
            }
            long centerMs = windowStartMs + Math.round((i + PAUSE_FRAMES / 2.0) * frameStepMs);
            double distSec = Math.abs(centerMs - candidateMs) / 1000.0;
            double score = pauseDb + DISTANCE_PENALTY_DB * distSec;
            if (score < bestScore) {
                bestScore = score;
                bestMs = centerMs;
            }
        }
        if (bestMs >= 0) {
            return bestMs;
        }
        return Math.max(0, candidateMs + (isStart ? fallbackCushionMs : -fallbackCushionMs));
    }

    private static long applyFallbackCushion(long candidateMs, boolean isStart,
                                             long durationMs, long fallbackCushionMs) {
        long shifted = Math.max(0, candidateMs + (isStart ? fallbackCushionMs : -fallbackCushionMs));
        return durationMs > 0 ? Math.min(durationMs, shifted) : shifted;
    }

    @Nullable
    private static float[] extractRmsEnvelope(@NonNull File file, long startMs, long endMs) {
        int numBuckets = (int) ((endMs - startMs) / FRAME_STEP_MS);
        if (numBuckets <= 0) {
            return null;
        }
        MediaExtractor extractor = null;
        MediaCodec codec = null;
        try {
            extractor = new MediaExtractor();
            extractor.setDataSource(file.getAbsolutePath());
            MediaFormat format = selectAudioTrack(extractor);
            if (format == null) {
                return null;
            }
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime == null) {
                return null;
            }
            extractor.seekTo(startMs * 1000L, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(format, null, null, 0);
            codec.start();
            return decodeToRms(extractor, codec, new PcmConfig(format), startMs, endMs, numBuckets);
        } catch (IOException | RuntimeException expected) {
            return null;
        } finally {
            releaseQuietly(codec, extractor);
        }
    }

    @Nullable
    private static MediaFormat selectAudioTrack(@NonNull MediaExtractor extractor) {
        int count = extractor.getTrackCount();
        for (int i = 0; i < count; i++) {
            MediaFormat trackFormat = extractor.getTrackFormat(i);
            String mime = trackFormat.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) {
                extractor.selectTrack(i);
                return trackFormat;
            }
        }
        return null;
    }

    @Nullable
    private static float[] decodeToRms(@NonNull MediaExtractor extractor, @NonNull MediaCodec codec,
                                       @NonNull PcmConfig config, long startMs, long endMs,
                                       int numBuckets) {
        double[] sumSq = new double[numBuckets];
        int[] counts = new int[numBuckets];
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false;
        long endUs = endMs * 1000L;
        for (int step = 0; step < MAX_CODEC_STEPS; step++) {
            if (!inputDone) {
                inputDone = queueNextSample(extractor, codec, endUs);
            }
            int outIndex = codec.dequeueOutputBuffer(info, CODEC_TIMEOUT_US);
            if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                config.update(codec.getOutputFormat());
            } else if (outIndex >= 0) {
                ByteBuffer outBuf = codec.getOutputBuffer(outIndex);
                if (outBuf != null && info.size > 0 && config.encoding == AudioFormat.ENCODING_PCM_16BIT) {
                    accumulatePcm16(outBuf, info, config, startMs, sumSq, counts);
                }
                codec.releaseOutputBuffer(outIndex, false);
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        || info.presentationTimeUs >= endUs) {
                    break;
                }
            } else if (inputDone) {
                break;
            }
        }
        return finalizeRms(sumSq, counts);
    }

    private static boolean queueNextSample(@NonNull MediaExtractor extractor, @NonNull MediaCodec codec,
                                           long endUs) {
        int inIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US);
        if (inIndex < 0) {
            return false;
        }
        ByteBuffer inBuf = codec.getInputBuffer(inIndex);
        if (inBuf == null) {
            return false;
        }
        long sampleTimeUs = extractor.getSampleTime();
        if (sampleTimeUs < 0 || sampleTimeUs > endUs + 200_000L) {
            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            return true;
        }
        int size = extractor.readSampleData(inBuf, 0);
        if (size < 0) {
            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
            return true;
        }
        codec.queueInputBuffer(inIndex, 0, size, sampleTimeUs, 0);
        extractor.advance();
        return false;
    }

    private static void accumulatePcm16(@NonNull ByteBuffer buffer, @NonNull MediaCodec.BufferInfo info,
                                        @NonNull PcmConfig config, long startMs,
                                        @NonNull double[] sumSq, @NonNull int[] counts) {
        ByteBuffer dup = buffer.duplicate();
        dup.position(info.offset);
        dup.limit(info.offset + info.size);
        dup.order(ByteOrder.LITTLE_ENDIAN);
        ShortBuffer shorts = dup.asShortBuffer();
        int totalFrames = shorts.remaining() / config.channels;
        long baseUs = info.presentationTimeUs;
        long startUs = startMs * 1000L;
        long stepUs = FRAME_STEP_MS * 1000L;
        for (int f = 0; f < totalFrames; f++) {
            long frameUs = baseUs + 1_000_000L * f / config.sampleRate;
            if (frameUs < startUs) {
                continue;
            }
            int bucket = (int) ((frameUs - startUs) / stepUs);
            if (bucket >= sumSq.length) {
                break;
            }
            double mono = 0;
            for (int c = 0; c < config.channels; c++) {
                mono += shorts.get(f * config.channels + c) / 32768.0;
            }
            mono /= config.channels;
            sumSq[bucket] += mono * mono;
            counts[bucket]++;
        }
    }

    @Nullable
    private static float[] finalizeRms(@NonNull double[] sumSq, @NonNull int[] counts) {
        int filled = 0;
        float[] rms = new float[sumSq.length];
        for (int i = 0; i < sumSq.length; i++) {
            if (counts[i] > 0) {
                rms[i] = (float) Math.sqrt(sumSq[i] / counts[i]);
                filled++;
            }
        }
        return filled >= PAUSE_FRAMES + 2 ? rms : null;
    }

    private static void releaseQuietly(@Nullable MediaCodec codec, @Nullable MediaExtractor extractor) {
        if (codec != null) {
            try {
                codec.stop();
            } catch (RuntimeException expected) {
            }
            try {
                codec.release();
            } catch (RuntimeException expected) {
            }
        }
        if (extractor != null) {
            try {
                extractor.release();
            } catch (RuntimeException expected) {
            }
        }
    }
}
