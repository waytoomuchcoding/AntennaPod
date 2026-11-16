package de.danoeh.antennapod.net.download.service.episode;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AudioPauseSnapperTest {
    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @Test
    public void testSnapToDeepSplicePauseOnStart() {
        float[] rms = new float[100];
        Arrays.fill(rms, 0.12f);
        for (int i = 44; i < 50; i++) {
            rms[i] = 0.0001f;
        }
        long snapped = AudioPauseSnapper.snapToPause(60_000L, true, 58_500L, rms, 50L, 1_200L);
        assertTrue(Math.abs(snapped - 60_850L) <= 100L);
    }

    @Test
    public void testPrefersCloserPauseWhenDropsEqual() {
        float[] rms = new float[100];
        Arrays.fill(rms, 0.15f);
        for (int i = 10; i < 15; i++) {
            rms[i] = 0.01f;
        }
        for (int i = 66; i < 71; i++) {
            rms[i] = 0.01f;
        }
        long snapped = AudioPauseSnapper.snapToPause(100_000L, false, 96_500L, rms, 50L, 1_200L);
        assertTrue(Math.abs(snapped - 99_925L) <= 100L);
    }

    @Test
    public void testAppliesInwardCushionWhenNoPausePresent() {
        float[] flatMusicRms = new float[100];
        Arrays.fill(flatMusicRms, 0.10f);
        long startSnapped = AudioPauseSnapper.snapToPause(80_000L, true, 78_500L, flatMusicRms, 50L, 1_200L);
        long endSnapped = AudioPauseSnapper.snapToPause(140_000L, false, 136_500L, flatMusicRms, 50L, 1_200L);
        assertEquals(81_200L, startSnapped);
        assertEquals(138_800L, endSnapped);
    }

    @Test
    public void testCoarseFallbackCushionAndEndpoints() throws IOException {
        File dummy = folder.newFile("unreadable.mp3");
        assertEquals(0L, AudioPauseSnapper.refineBoundary(dummy, 0L, true, 600_000L, 4_000L));
        assertEquals(600_000L, AudioPauseSnapper.refineBoundary(dummy, 600_000L, false, 600_000L, 4_000L));
        assertEquals(104_000L, AudioPauseSnapper.refineBoundary(dummy, 100_000L, true, 600_000L, 4_000L));
        assertEquals(176_000L, AudioPauseSnapper.refineBoundary(dummy, 180_000L, false, 600_000L, 4_000L));
    }
}
