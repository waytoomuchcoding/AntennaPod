package de.danoeh.antennapod.net.download.service.episode;

import android.media.MediaFormat;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.shadows.ShadowMediaExtractor;
import org.robolectric.shadows.util.DataSource;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

@RunWith(RobolectricTestRunner.class)
public class AacClipExtractorTest {
    private static final long FRAME_US = 1024L * 1_000_000 / 44100;
    private static final int FRAME_SIZE = 300;

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @After
    public void tearDown() {
        ShadowMediaExtractor.reset();
    }

    private static class FakeAacStream implements AacClipExtractor.SampleReader {
        private final int numFrames;
        private int index = 0;

        FakeAacStream(int numFrames) {
            this.numFrames = numFrames;
        }

        @Override
        public void seekTo(long timeUs) {
            index = (int) Math.min(numFrames, timeUs / FRAME_US);
        }

        @Override
        public long getSampleTime() {
            return index < numFrames ? index * FRAME_US : -1;
        }

        @Override
        public int readSampleData(ByteBuffer buffer) {
            if (index >= numFrames) {
                return -1;
            }
            buffer.put(new byte[FRAME_SIZE]);
            return FRAME_SIZE;
        }

        @Override
        public void advance() {
            index++;
        }
    }

    @Test
    public void testCopySamplesOfClip() {
        List<Long> times = new ArrayList<>();
        long firstUs = AacClipExtractor.copySamples(new FakeAacStream(60_000),
                (buffer, size, presentationTimeUs) -> times.add(presentationTimeUs),
                600_000_000L, 1_200_000_000L, 4096);
        long firstFrame = 600_000_000L / FRAME_US;
        assertEquals(firstFrame * FRAME_US, firstUs);
        assertEquals(0, (long) times.get(0));
        assertEquals(FRAME_US, (long) times.get(1));
        long lastFrame = (1_200_000_000L - 1) / FRAME_US;
        assertEquals(lastFrame - firstFrame + 1, times.size());
        assertEquals((lastFrame - firstFrame) * FRAME_US, (long) times.get(times.size() - 1));
    }

    @Test
    public void testCopySamplesStopsAtEndOfStream() {
        List<Long> times = new ArrayList<>();
        long firstUs = AacClipExtractor.copySamples(new FakeAacStream(100),
                (buffer, size, presentationTimeUs) -> times.add(presentationTimeUs),
                0, 1_200_000_000L, 4096);
        assertEquals(0, firstUs);
        assertEquals(100, times.size());
    }

    @Test
    public void testCopySamplesAfterEndOfStream() {
        List<Long> times = new ArrayList<>();
        long firstUs = AacClipExtractor.copySamples(new FakeAacStream(100),
                (buffer, size, presentationTimeUs) -> times.add(presentationTimeUs),
                600_000_000L, 1_200_000_000L, 4096);
        assertEquals(-1, firstUs);
        assertEquals(0, times.size());
    }

    @Test
    public void testOpenAacFile() throws IOException {
        File file = folder.newFile("episode.m4a");
        MediaFormat format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, 44100, 2);
        format.setLong(MediaFormat.KEY_DURATION, 1_800_000_000L);
        ShadowMediaExtractor.addTrack(DataSource.toDataSource(file.getAbsolutePath()), format, new byte[1000]);
        try (AacClipExtractor extractor = AacClipExtractor.open(file)) {
            assertNotNull(extractor);
            assertEquals(1_800_000, extractor.getDurationMs());
        }
    }

    @Test
    public void testOpenNonAacFile() throws IOException {
        File file = folder.newFile("episode.ogg");
        MediaFormat format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_OPUS, 48000, 2);
        ShadowMediaExtractor.addTrack(DataSource.toDataSource(file.getAbsolutePath()), format, new byte[1000]);
        assertNull(AacClipExtractor.open(file));
    }
}
