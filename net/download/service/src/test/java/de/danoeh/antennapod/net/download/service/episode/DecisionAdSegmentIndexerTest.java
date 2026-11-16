package de.danoeh.antennapod.net.download.service.episode;

import de.danoeh.antennapod.model.feed.AdSegment;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class DecisionAdSegmentIndexerTest {

    @Test
    public void testViterbiSmoothSuppressesIsolatedBlipAndBridgesDip() {
        double[] probs = {0.02, 0.51, 0.03, 0.94, 0.25, 0.92, 0.02, 0.85, 0.03};
        boolean[] active = DecisionAdSegmentIndexer.viterbiSmooth(probs);
        assertFalse(active[0]);
        assertFalse(active[1]);
        assertFalse(active[2]);
        assertTrue(active[3]);
        assertTrue(active[4]);
        assertTrue(active[5]);
        assertFalse(active[6]);
        assertTrue(active[7]);
        assertFalse(active[8]);
    }

    @Test
    public void testBuildCoarseSegmentsSeparatesAdAndSelfPromo() {
        List<Long> starts = Arrays.asList(0L, 20_000L, 40_000L, 60_000L, 80_000L, 100_000L);
        List<Long> ends = Arrays.asList(20_000L, 40_000L, 60_000L, 80_000L, 100_000L, 120_000L);
        DecisionAdSegmentIndexer.BlockScore[] scores = {
            new DecisionAdSegmentIndexer.BlockScore(0.95, 0.02),
            new DecisionAdSegmentIndexer.BlockScore(0.04, 0.85),
            new DecisionAdSegmentIndexer.BlockScore(0.03, 0.88),
            new DecisionAdSegmentIndexer.BlockScore(0.01, 0.01),
            new DecisionAdSegmentIndexer.BlockScore(0.92, 0.03),
            new DecisionAdSegmentIndexer.BlockScore(0.02, 0.01)
        };
        List<DecisionAdSegmentIndexer.CandidateSegment> segs =
                DecisionAdSegmentIndexer.buildCoarseSegments(starts, ends, scores);
        assertEquals(3, segs.size());
        assertEquals(AdSegment.TYPE_AD, segs.get(0).type);
        assertEquals(0L, segs.get(0).startMs);
        assertEquals(20_000L, segs.get(0).endMs);
        assertEquals(AdSegment.TYPE_SELF_PROMO, segs.get(1).type);
        assertEquals(20_000L, segs.get(1).startMs);
        assertEquals(60_000L, segs.get(1).endMs);
        assertEquals(AdSegment.TYPE_AD, segs.get(2).type);
        assertEquals(80_000L, segs.get(2).startMs);
        assertEquals(100_000L, segs.get(2).endMs);
    }

    @Test
    public void testAsymmetricChangepointProtectsMixedBoundarySlice() {
        List<Long> starts = Arrays.asList(10_000L, 13_000L, 16_000L, 19_000L, 22_000L);
        List<Long> ends = Arrays.asList(13_000L, 16_000L, 19_000L, 22_000L, 25_000L);
        DecisionAdSegmentIndexer.BlockScore[] startScores = {
            new DecisionAdSegmentIndexer.BlockScore(0.02, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.52, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.93, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.96, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.95, 0.0)
        };
        assertEquals(16_000L, DecisionAdSegmentIndexer.fitBoundaryChangepoint(starts, ends, startScores, true));

        DecisionAdSegmentIndexer.BlockScore[] endScores = {
            new DecisionAdSegmentIndexer.BlockScore(0.95, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.92, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.54, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.03, 0.0),
            new DecisionAdSegmentIndexer.BlockScore(0.01, 0.0)
        };
        assertEquals(16_000L, DecisionAdSegmentIndexer.fitBoundaryChangepoint(starts, ends, endScores, false));
    }

    @Test
    public void testInternalTypeChangepointFindsAdToSelfPromoTransition() {
        List<Long> starts = Arrays.asList(10_000L, 13_000L, 16_000L, 19_000L);
        List<Long> ends = Arrays.asList(13_000L, 16_000L, 19_000L, 22_000L);
        DecisionAdSegmentIndexer.BlockScore[] scores = {
            new DecisionAdSegmentIndexer.BlockScore(0.92, 0.04),
            new DecisionAdSegmentIndexer.BlockScore(0.89, 0.06),
            new DecisionAdSegmentIndexer.BlockScore(0.08, 0.84),
            new DecisionAdSegmentIndexer.BlockScore(0.03, 0.91)
        };
        assertEquals(16_000L, DecisionAdSegmentIndexer.fitInternalTypeChangepoint(
                starts, ends, scores, AdSegment.TYPE_AD));
    }

    @Test
    public void testParseRetryDelayMs() {
        assertEquals(47_000L, DecisionAdSegmentIndexer.parseRetryDelayMs(
                "Please retry in 45.5s.", 35_000L));
        assertEquals(23_500L, DecisionAdSegmentIndexer.parseRetryDelayMs(
                "{\"retryDelay\": \"22s\"}", 35_000L));
        assertEquals(35_000L, DecisionAdSegmentIndexer.parseRetryDelayMs(
                "rate limit without delay", 35_000L));
        assertEquals(0L, DecisionAdSegmentIndexer.parseRetryDelayMs(
                "Please retry in 45s", 0L));
    }
}
