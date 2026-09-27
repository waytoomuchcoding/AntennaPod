package de.danoeh.antennapod.model.feed;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class AdSegmentTest {
    private static final List<AdSegment> SEGMENTS = Arrays.asList(
            new AdSegment(100_000, 160_000), new AdSegment(300_000, 330_000));

    @Test
    public void testSkipForwardWithoutAds() {
        assertEquals(40_000, AdSegment.skipForward(null, 10_000, 30_000));
        assertEquals(40_000, AdSegment.skipForward(SEGMENTS, 10_000, 30_000));
    }

    @Test
    public void testSkipForwardOverAd() {
        assertEquals(180_000, AdSegment.skipForward(SEGMENTS, 90_000, 30_000));
        assertEquals(160_000, AdSegment.skipForward(SEGMENTS, 70_000, 30_000));
    }

    @Test
    public void testSkipForwardOverMultipleAds() {
        assertEquals(340_000, AdSegment.skipForward(SEGMENTS, 90_000, 160_000));
    }

    @Test
    public void testSkipForwardFromInsideAd() {
        assertEquals(190_000, AdSegment.skipForward(SEGMENTS, 120_000, 30_000));
    }

    @Test
    public void testSkipBackwardWithoutAds() {
        assertEquals(0, AdSegment.skipBackward(null, 5_000, 10_000));
        assertEquals(80_000, AdSegment.skipBackward(SEGMENTS, 90_000, 10_000));
    }

    @Test
    public void testSkipBackwardOverAd() {
        assertEquals(95_000, AdSegment.skipBackward(SEGMENTS, 165_000, 10_000));
        assertEquals(160_000, AdSegment.skipBackward(SEGMENTS, 170_000, 10_000));
        assertEquals(90_000, AdSegment.skipBackward(SEGMENTS, 160_000, 10_000));
    }

    @Test
    public void testGetSegmentAt() {
        assertEquals(100_000, AdSegment.getSegmentAt(SEGMENTS, 100_000).getStart());
        assertNull(AdSegment.getSegmentAt(SEGMENTS, 160_000));
        assertNull(AdSegment.getSegmentAt(null, 120_000));
    }

    @Test
    public void testMerge() {
        List<AdSegment> merged = AdSegment.merge(Arrays.asList(new AdSegment(50_000, 60_000),
                new AdSegment(10_000, 20_000), new AdSegment(22_000, 30_000), new AdSegment(40_000, 40_000)), 5_000);
        assertEquals(2, merged.size());
        assertEquals(10_000, merged.get(0).getStart());
        assertEquals(30_000, merged.get(0).getEnd());
        assertEquals(50_000, merged.get(1).getStart());
    }

    @Test
    public void testMergeKeepsTypesSeparate() {
        List<AdSegment> merged = AdSegment.merge(Arrays.asList(new AdSegment(0, 10_000),
                new AdSegment(12_000, 20_000, AdSegment.TYPE_SELF_PROMO, true),
                new AdSegment(15_000, 30_000, AdSegment.TYPE_SELF_PROMO, true)), 5_000);
        assertEquals(2, merged.size());
        assertEquals(AdSegment.TYPE_SELF_PROMO, merged.get(1).getType());
        assertEquals(12_000, merged.get(1).getStart());
        assertEquals(30_000, merged.get(1).getEnd());
    }

    @Test
    public void testGetSkippable() {
        List<AdSegment> segments = Arrays.asList(new AdSegment(0, 10_000),
                new AdSegment(20_000, 30_000, AdSegment.TYPE_SELF_PROMO, true),
                new AdSegment(40_000, 50_000, AdSegment.TYPE_AD, false));
        assertEquals(2, AdSegment.getSkippable(segments, true).size());
        assertEquals(1, AdSegment.getSkippable(segments, false).size());
        assertNull(AdSegment.getSkippable(null, true));
    }

    @Test
    public void testSerializeAndParseTypeAndEnabled() {
        List<AdSegment> parsed = AdSegment.parse(AdSegment.serialize(Arrays.asList(
                new AdSegment(20_000, 30_000, AdSegment.TYPE_SELF_PROMO, false))));
        assertEquals(AdSegment.TYPE_SELF_PROMO, parsed.get(0).getType());
        assertFalse(parsed.get(0).isEnabled());
        assertNull(AdSegment.parse(null));
        assertTrue(AdSegment.parse("").isEmpty());
    }

    @Test
    public void testSerializeAndParse() {
        List<AdSegment> parsed = AdSegment.parse(AdSegment.serialize(SEGMENTS) + "invalid\n");
        assertEquals(2, parsed.size());
        assertEquals(300_000, parsed.get(1).getStart());
        assertEquals(330_000, parsed.get(1).getEnd());
    }
}
