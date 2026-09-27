package de.danoeh.antennapod.net.download.service.episode;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AdSegmentIndexerTest {
    private static final long CLIP_MS = 600_000;
    private static final String TEN_WORDS = "one two three four five six seven eight nine ten";

    private static List<AdSegmentIndexer.TranscriptLine> transcript(long offsetMs, int numLines) {
        List<AdSegmentIndexer.TranscriptLine> lines = new ArrayList<>();
        for (int i = 0; i < numLines && offsetMs + i * 4000L < CLIP_MS; i++) {
            lines.add(new AdSegmentIndexer.TranscriptLine(offsetMs + i * 4000L, TEN_WORDS));
        }
        return lines;
    }

    @Test
    public void testTimingMisfitOfConsistentTranscript() {
        assertEquals(0, AdSegmentIndexer.getTimingMisfit(transcript(0, 150), CLIP_MS), 0.001);
    }

    @Test
    public void testTimingMisfitOfShiftedTranscript() {
        double consistent = AdSegmentIndexer.getTimingMisfit(transcript(0, 150), CLIP_MS);
        double shifted = AdSegmentIndexer.getTimingMisfit(transcript(106_000, 150), CLIP_MS);
        assertTrue(shifted > consistent + 0.1);
    }

    @Test
    public void testTimingMisfitOfTranscriptWithGap() {
        double consistent = AdSegmentIndexer.getTimingMisfit(transcript(0, 150), CLIP_MS);
        double gap = AdSegmentIndexer.getTimingMisfit(transcript(0, 40), CLIP_MS);
        assertTrue(gap > consistent + 0.5);
    }

    @Test
    public void testTimingMisfitOfEmptyTranscript() {
        assertEquals(Double.MAX_VALUE, AdSegmentIndexer.getTimingMisfit(new ArrayList<>(), CLIP_MS), 0);
    }
}
