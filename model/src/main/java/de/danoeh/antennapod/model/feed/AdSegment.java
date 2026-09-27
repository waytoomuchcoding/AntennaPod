package de.danoeh.antennapod.model.feed;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AdSegment {
    public static final int TYPE_AD = 0;
    public static final int TYPE_SELF_PROMO = 1;

    private final long start;
    private final long end;
    private final int type;
    private final boolean enabled;

    public AdSegment(long start, long end) {
        this(start, end, TYPE_AD, true);
    }

    public AdSegment(long start, long end, int type, boolean enabled) {
        this.start = start;
        this.end = end;
        this.type = type;
        this.enabled = enabled;
    }

    public long getStart() {
        return start;
    }

    public long getEnd() {
        return end;
    }

    public int getType() {
        return type;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public long getDuration() {
        return end - start;
    }

    @NonNull
    public AdSegment withEnabled(boolean enabled) {
        return new AdSegment(start, end, type, enabled);
    }

    @NonNull
    public static List<AdSegment> merge(@NonNull List<AdSegment> segments, long maxGap) {
        List<AdSegment> sorted = new ArrayList<>();
        for (AdSegment segment : segments) {
            if (segment.end > segment.start) {
                sorted.add(segment);
            }
        }
        Collections.sort(sorted, (a, b) -> Long.compare(a.start, b.start));
        List<AdSegment> result = new ArrayList<>();
        for (AdSegment segment : sorted) {
            AdSegment last = result.isEmpty() ? null : result.get(result.size() - 1);
            if (last != null && last.type == segment.type && segment.start <= last.end + maxGap) {
                result.set(result.size() - 1, new AdSegment(last.start, Math.max(last.end, segment.end),
                        last.type, last.enabled && segment.enabled));
            } else if (last != null && segment.start < last.end) {
                if (segment.end > last.end) {
                    result.add(new AdSegment(last.end, segment.end, segment.type, segment.enabled));
                }
            } else {
                result.add(segment);
            }
        }
        return result;
    }

    @Nullable
    public static List<AdSegment> getSkippable(@Nullable List<AdSegment> segments, boolean includeSelfPromos) {
        if (segments == null) {
            return null;
        }
        List<AdSegment> result = new ArrayList<>();
        for (AdSegment segment : segments) {
            if (segment.enabled && (segment.type == TYPE_AD || includeSelfPromos)) {
                result.add(segment);
            }
        }
        return result;
    }

    @Nullable
    public static AdSegment getSegmentAt(@Nullable List<AdSegment> segments, long position) {
        if (segments == null) {
            return null;
        }
        for (AdSegment segment : segments) {
            if (segment.start <= position && position < segment.end) {
                return segment;
            }
        }
        return null;
    }

    public static long skipForward(@Nullable List<AdSegment> segments, long position, long delta) {
        if (segments == null) {
            return position + delta;
        }
        long target = position;
        long remaining = delta;
        for (AdSegment segment : segments) {
            if (segment.end <= target) {
                continue;
            }
            if (segment.start <= target) {
                target = segment.end;
                continue;
            }
            if (target + remaining < segment.start) {
                break;
            }
            remaining -= segment.start - target;
            target = segment.end;
        }
        return target + remaining;
    }

    public static long skipBackward(@Nullable List<AdSegment> segments, long position, long delta) {
        if (segments == null) {
            return Math.max(0, position - delta);
        }
        long target = position;
        long remaining = delta;
        for (int i = segments.size() - 1; i >= 0; i--) {
            AdSegment segment = segments.get(i);
            if (segment.start >= target) {
                continue;
            }
            if (segment.end >= target) {
                target = segment.start;
                continue;
            }
            if (target - remaining >= segment.end) {
                break;
            }
            remaining -= target - segment.end;
            target = segment.start;
        }
        return Math.max(0, target - remaining);
    }

    @NonNull
    public static String serialize(@NonNull List<AdSegment> segments) {
        StringBuilder builder = new StringBuilder();
        for (AdSegment segment : segments) {
            builder.append(segment.start).append(' ').append(segment.end).append(' ')
                    .append(segment.type).append(' ').append(segment.enabled ? 1 : 0).append('\n');
        }
        return builder.toString();
    }

    @Nullable
    public static List<AdSegment> parse(@Nullable String str) {
        if (str == null) {
            return null;
        }
        List<AdSegment> segments = new ArrayList<>();
        for (String line : str.split("\n")) {
            String[] parts = line.trim().split(" ");
            if (parts.length != 2 && parts.length != 4) {
                continue;
            }
            try {
                long start = Long.parseLong(parts[0]);
                long end = Long.parseLong(parts[1]);
                if (parts.length == 4) {
                    segments.add(new AdSegment(start, end, Integer.parseInt(parts[2]), !"0".equals(parts[3])));
                } else {
                    segments.add(new AdSegment(start, end));
                }
            } catch (NumberFormatException expected) {
            }
        }
        Collections.sort(segments, (a, b) -> Long.compare(a.start, b.start));
        return segments;
    }
}
