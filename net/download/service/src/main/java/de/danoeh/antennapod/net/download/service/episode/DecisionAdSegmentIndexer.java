package de.danoeh.antennapod.net.download.service.episode;

import android.util.Log;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import de.danoeh.antennapod.model.feed.AdSegment;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okio.ByteString;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DecisionAdSegmentIndexer {
    private static final String TAG = "DecisionAdIndexer";
    private static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com";
    private static final String MODEL = "gemini-decision-1";
    private static final MediaType JSON_MEDIA = MediaType.get("application/json; charset=utf-8");

    static final long COARSE_BLOCK_MS = 20_000L;
    static final long FINE_WINDOW_HALF_MS = 21_000L;
    static final long FINE_STEP_MS = 3_000L;
    static final long REFINED_CUSHION_MS = 1_200L;
    static final long COARSE_CUSHION_MS = 4_000L;

    private static final int MAX_BLOCKS_PER_BATCH = 30;
    private static final long MAX_BATCH_RAW_BYTES = 10L * 1024L * 1024L;
    private static final int MAX_EDGES_PER_BATCH = 6;
    private static final long MAX_URI_DURATION_MS = 20L * 60L * 1000L;
    private static final long ENDPOINT_MARGIN_MS = 2_000L;
    private static final long ADJACENT_TOLERANCE_MS = 500L;
    private static final long MIN_SEGMENT_MS = 8_000L;
    private static final long MERGE_GAP_MS = 5_000L;
    private static final int MIN_BISECT_LEAF = 3;

    private static final double VITERBI_ALPHA = 2.0;
    private static final double VITERBI_SWITCH = 0.15;
    private static final double VITERBI_BIAS = 0.55;
    private static final double CONTENT_PROTECTION_PENALTY = 1.85;

    private static final int MAX_SERVER_RETRIES = 4;
    private static final int MAX_RATE_LIMIT_RETRIES = 4;
    private static final int MAX_HARD_QUOTA_RETRIES = 2;
    private static final long DEFAULT_RETRY_DELAY_MS = 35_000L;
    private static final Pattern RETRY_DELAY_PATTERN = Pattern.compile(
            "(?:retry in |\"retryDelay\"\\s*:\\s*\")([0-9]+(?:\\.[0-9]+)?)s",
            Pattern.CASE_INSENSITIVE);

    private final OkHttpClient client;
    private final String apiKey;
    private final File tempDir;
    private final String baseUrl;
    private final long baseRetryDelayMs;
    private AdSegmentIndexer.ProgressListener progressListener = percent -> { };

    interface SliceSource {
        @NonNull
        List<AudioSlice> sliceRange(long startMs, long endMs, long stepMs) throws IOException;
    }

    static final class AudioSlice {
        final long startMs;
        final long endMs;
        final String mimeType;
        final String base64Data;
        final int rawByteLength;

        AudioSlice(long startMs, long endMs, @NonNull String mimeType,
                   @NonNull String base64Data, int rawByteLength) {
            this.startMs = startMs;
            this.endMs = endMs;
            this.mimeType = mimeType;
            this.base64Data = base64Data;
            this.rawByteLength = rawByteLength;
        }
    }

    static final class BlockScore {
        final double adProb;
        final double selfPromoProb;

        BlockScore(double adProb, double selfPromoProb) {
            this.adProb = adProb;
            this.selfPromoProb = selfPromoProb;
        }

        double nonContentProb() {
            return Math.min(0.999, Math.max(0.001, adProb + selfPromoProb));
        }
    }

    static final class CandidateSegment {
        long startMs;
        long endMs;
        int type;
        boolean startRefined;
        boolean endRefined;

        CandidateSegment(long startMs, long endMs, int type) {
            this.startMs = startMs;
            this.endMs = endMs;
            this.type = type;
        }
    }

    private static final class EdgeTask {
        final int segmentIndex;
        final boolean isStart;
        final boolean internalTransition;
        final long centerMs;
        final List<AudioSlice> slices;

        EdgeTask(int segmentIndex, boolean isStart, boolean internalTransition,
                 long centerMs, @NonNull List<AudioSlice> slices) {
            this.segmentIndex = segmentIndex;
            this.isStart = isStart;
            this.internalTransition = internalTransition;
            this.centerMs = centerMs;
            this.slices = slices;
        }

        long totalRawBytes() {
            long total = 0;
            for (AudioSlice slice : slices) {
                total += slice.rawByteLength;
            }
            return total;
        }
    }

    private static final class ProhibitedContentException extends IOException {
        ProhibitedContentException(String message) {
            super(message);
        }
    }

    public DecisionAdSegmentIndexer(@NonNull OkHttpClient client, @NonNull String apiKey,
                                    @NonNull File tempDir) {
        this(client, apiKey, tempDir, DEFAULT_BASE_URL, DEFAULT_RETRY_DELAY_MS);
    }

    DecisionAdSegmentIndexer(@NonNull OkHttpClient client, @NonNull String apiKey,
                             @NonNull File tempDir, @NonNull String baseUrl, long baseRetryDelayMs) {
        this.client = client.newBuilder()
                .readTimeout(5, TimeUnit.MINUTES)
                .writeTimeout(5, TimeUnit.MINUTES)
                .build();
        this.apiKey = apiKey;
        this.tempDir = tempDir;
        this.baseUrl = baseUrl;
        this.baseRetryDelayMs = baseRetryDelayMs;
    }

    public void setProgressListener(@NonNull AdSegmentIndexer.ProgressListener progressListener) {
        this.progressListener = progressListener;
    }

    @NonNull
    public List<AdSegment> index(@NonNull File file, @Nullable String mimeType, long durationMs,
                                 @Nullable String episodeTitle, @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        String uploadMime = AdSegmentIndexer.getUploadMimeType(file, mimeType);
        List<Mp3FrameSplitter.Chunk> mp3Chunks = "audio/mpeg".equals(uploadMime)
                ? Mp3FrameSplitter.split(file, COARSE_BLOCK_MS) : new ArrayList<>();
        if (!mp3Chunks.isEmpty()) {
            long totalMs = durationMs > 0 ? durationMs : mp3Chunks.get(mp3Chunks.size() - 1).endMs;
            List<AudioSlice> coarseSlices = readMp3Slices(file, mp3Chunks);
            SliceSource mp3Source = (s, e, step) -> readMp3Slices(file,
                    Mp3FrameSplitter.splitRange(file, s, e, step));
            return indexFromSlices(file, coarseSlices, mp3Source, totalMs, episodeTitle, podcastTitle);
        }
        try (AacClipExtractor aacExtractor = AacClipExtractor.open(file)) {
            if (aacExtractor != null) {
                return indexAacFile(file, aacExtractor, durationMs, episodeTitle, podcastTitle);
            }
        }
        if (durationMs > 0 && durationMs <= MAX_URI_DURATION_MS) {
            return indexViaUploadedUri(file, uploadMime, durationMs, episodeTitle, podcastTitle);
        }
        AdSegmentIndexer fallback = new AdSegmentIndexer(client, apiKey, tempDir);
        fallback.setProgressListener(progressListener);
        return fallback.index(file, mimeType, durationMs, episodeTitle, podcastTitle);
    }

    @NonNull
    private List<AdSegment> indexAacFile(@NonNull File file, @NonNull AacClipExtractor extractor,
                                         long durationMs, @Nullable String episodeTitle,
                                         @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        long totalMs = durationMs > 0 ? durationMs : extractor.getDurationMs();
        if (totalMs <= 0) {
            return new ArrayList<>();
        }
        File clipFile = File.createTempFile("ad_decision", ".m4a", tempDir);
        try {
            SliceSource aacSource = (startMs, endMs, stepMs) -> extractAacSlices(
                    extractor, clipFile, startMs, endMs, stepMs);
            List<AudioSlice> coarseSlices = aacSource.sliceRange(0, totalMs, COARSE_BLOCK_MS);
            return indexFromSlices(file, coarseSlices, aacSource, totalMs, episodeTitle, podcastTitle);
        } finally {
            if (clipFile.exists() && !clipFile.delete()) {
                Log.d(TAG, "Unable to delete temporary clip");
            }
        }
    }

    @NonNull
    List<AdSegment> indexFromSlices(@NonNull File file, @NonNull List<AudioSlice> coarseSlices,
                                    @NonNull SliceSource fineSource, long totalDurationMs,
                                    @Nullable String episodeTitle, @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        if (coarseSlices.isEmpty()) {
            return new ArrayList<>();
        }
        String rubric = buildRubric(episodeTitle, podcastTitle);
        BlockScore[] coarseScores = scoreCoarsePass(coarseSlices, rubric);
        List<Long> starts = new ArrayList<>(coarseSlices.size());
        List<Long> ends = new ArrayList<>(coarseSlices.size());
        for (AudioSlice slice : coarseSlices) {
            starts.add(slice.startMs);
            ends.add(slice.endMs);
        }
        List<CandidateSegment> candidates = buildCoarseSegments(starts, ends, coarseScores);
        if (candidates.isEmpty()) {
            progressListener.onProgress(100);
            return new ArrayList<>();
        }
        refineBoundaries(candidates, fineSource, totalDurationMs, rubric);
        List<AdSegment> segments = snapAndFinalizeSegments(file, candidates, totalDurationMs);
        progressListener.onProgress(100);
        return segments;
    }

    @NonNull
    private List<AdSegment> indexViaUploadedUri(@NonNull File file, @NonNull String uploadMime,
                                                long durationMs, @Nullable String episodeTitle,
                                                @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        AdSegmentIndexer uploader = new AdSegmentIndexer(client, apiKey, tempDir);
        JSONObject uploaded = uploader.upload(file, 0, file.length(), uploadMime);
        try {
            uploaded = uploader.waitUntilActive(uploaded);
            progressListener.onProgress(35);
            String uri = uploaded.getString("uri");
            String rubric = buildRubric(episodeTitle, podcastTitle);
            int numBlocks = (int) Math.max(1, (durationMs + COARSE_BLOCK_MS - 1) / COARSE_BLOCK_MS);
            List<Long> starts = new ArrayList<>(numBlocks);
            List<Long> ends = new ArrayList<>(numBlocks);
            JSONArray input = new JSONArray()
                    .put(new JSONObject().put("type", "text").put("text", rubric))
                    .put(new JSONObject().put("type", "audio")
                            .put("mime_type", uploadMime).put("uri", uri));
            JSONObject questions = new JSONObject();
            JSONObject criteria = buildChoiceCriteria();
            for (int b = 0; b < numBlocks; b++) {
                long s = b * COARSE_BLOCK_MS;
                long e = Math.min(durationMs, (b + 1) * COARSE_BLOCK_MS);
                starts.add(s);
                ends.add(e);
                questions.put("b" + b, new JSONObject()
                        .put("type", "choice")
                        .put("instructions", "Classify segment " + formatTimestamp(s)
                                + "-" + formatTimestamp(e) + ".")
                        .put("criteria", criteria));
            }
            JSONObject answers = executeDecisions(input, questions);
            progressListener.onProgress(85);
            BlockScore[] scores = new BlockScore[numBlocks];
            for (int b = 0; b < numBlocks; b++) {
                scores[b] = parseAnswerScore(answers.optJSONObject("b" + b));
            }
            List<CandidateSegment> candidates = buildCoarseSegments(starts, ends, scores);
            List<AdSegment> result = snapAndFinalizeSegments(file, candidates, durationMs);
            progressListener.onProgress(100);
            return result;
        } finally {
            uploader.deleteUploadedFile(uploaded.getString("name"));
        }
    }

    @NonNull
    private BlockScore[] scoreCoarsePass(@NonNull List<AudioSlice> slices, @NonNull String rubric)
            throws IOException, JSONException, InterruptedException {
        BlockScore[] scores = new BlockScore[slices.size()];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = new BlockScore(0, 0);
        }
        int idx = 0;
        while (idx < slices.size()) {
            int batchEnd = idx;
            long batchBytes = 0;
            while (batchEnd < slices.size()
                    && batchEnd - idx < MAX_BLOCKS_PER_BATCH
                    && (batchEnd == idx
                    || batchBytes + slices.get(batchEnd).rawByteLength <= MAX_BATCH_RAW_BYTES)) {
                batchBytes += slices.get(batchEnd).rawByteLength;
                batchEnd++;
            }
            scoreBatchWithBisection(slices, idx, batchEnd, rubric, scores);
            idx = batchEnd;
            progressListener.onProgress((int) (75L * idx / slices.size()));
        }
        return scores;
    }

    private void scoreBatchWithBisection(@NonNull List<AudioSlice> slices, int from, int to,
                                         @NonNull String rubric, @NonNull BlockScore[] outScores)
            throws IOException, JSONException, InterruptedException {
        try {
            scoreSliceRange(slices, from, to, rubric, outScores);
        } catch (ProhibitedContentException e) {
            int count = to - from;
            if (count <= MIN_BISECT_LEAF) {
                Log.d(TAG, "Skipping prohibited story window B" + from + "..B" + (to - 1));
                return;
            }
            int mid = from + count / 2;
            scoreBatchWithBisection(slices, from, mid, rubric, outScores);
            scoreBatchWithBisection(slices, mid, to, rubric, outScores);
        }
    }

    private void scoreSliceRange(@NonNull List<AudioSlice> slices, int from, int to,
                                 @NonNull String rubric, @NonNull BlockScore[] outScores)
            throws IOException, JSONException, InterruptedException {
        JSONArray input = new JSONArray().put(new JSONObject()
                .put("type", "text")
                .put("text", rubric + "\nSequential audio blocks B" + from + "..B" + (to - 1) + ":"));
        JSONObject questions = new JSONObject();
        JSONObject criteria = buildChoiceCriteria();
        for (int b = from; b < to; b++) {
            AudioSlice slice = slices.get(b);
            input.put(new JSONObject().put("type", "text")
                    .put("text", "\nB" + b + " (" + formatTimestamp(slice.startMs)
                            + "-" + formatTimestamp(slice.endMs) + "):"));
            input.put(new JSONObject().put("type", "audio")
                    .put("mime_type", slice.mimeType)
                    .put("data", slice.base64Data));
            questions.put("b" + b, new JSONObject()
                    .put("type", "choice")
                    .put("instructions", "Classify B" + b + ".")
                    .put("criteria", criteria));
        }
        JSONObject answers = executeDecisions(input, questions);
        for (int b = from; b < to; b++) {
            outScores[b] = parseAnswerScore(answers.optJSONObject("b" + b));
        }
    }

    private void refineBoundaries(@NonNull List<CandidateSegment> candidates,
                                  @NonNull SliceSource fineSource, long totalDurationMs,
                                  @NonNull String rubric)
            throws IOException, JSONException, InterruptedException {
        List<EdgeTask> tasks = collectEdgeTasks(candidates, fineSource, totalDurationMs);
        if (tasks.isEmpty()) {
            progressListener.onProgress(92);
            return;
        }
        int idx = 0;
        while (idx < tasks.size()) {
            int batchEnd = idx;
            long batchBytes = 0;
            while (batchEnd < tasks.size()
                    && batchEnd - idx < MAX_EDGES_PER_BATCH
                    && (batchEnd == idx
                    || batchBytes + tasks.get(batchEnd).totalRawBytes() <= MAX_BATCH_RAW_BYTES)) {
                batchBytes += tasks.get(batchEnd).totalRawBytes();
                batchEnd++;
            }
            refineEdgeBatch(candidates, tasks.subList(idx, batchEnd), rubric);
            idx = batchEnd;
            progressListener.onProgress(75 + (int) (17L * idx / tasks.size()));
        }
    }

    @NonNull
    private List<EdgeTask> collectEdgeTasks(@NonNull List<CandidateSegment> candidates,
                                            @NonNull SliceSource fineSource, long totalDurationMs)
            throws IOException {
        List<EdgeTask> tasks = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            CandidateSegment cur = candidates.get(i);
            boolean prevAdjacent = i > 0
                    && Math.abs(candidates.get(i - 1).endMs - cur.startMs) <= ADJACENT_TOLERANCE_MS;
            boolean nextAdjacent = i + 1 < candidates.size()
                    && Math.abs(candidates.get(i + 1).startMs - cur.endMs) <= ADJACENT_TOLERANCE_MS;
            if (prevAdjacent && candidates.get(i - 1).type != cur.type) {
                addEdgeTaskIfNonEmpty(tasks, fineSource, i, true, true, cur.startMs, totalDurationMs);
            } else if (!prevAdjacent && cur.startMs > ENDPOINT_MARGIN_MS) {
                addEdgeTaskIfNonEmpty(tasks, fineSource, i, true, false, cur.startMs, totalDurationMs);
            }
            if (!nextAdjacent && (totalDurationMs <= 0
                    || cur.endMs < totalDurationMs - ENDPOINT_MARGIN_MS)) {
                addEdgeTaskIfNonEmpty(tasks, fineSource, i, false, false, cur.endMs, totalDurationMs);
            }
        }
        return tasks;
    }

    private void addEdgeTaskIfNonEmpty(@NonNull List<EdgeTask> tasks, @NonNull SliceSource fineSource,
                                       int segIndex, boolean isStart, boolean internalTransition,
                                       long centerMs, long totalDurationMs) throws IOException {
        long w0 = Math.max(0, centerMs - FINE_WINDOW_HALF_MS);
        long rawW1 = centerMs + FINE_WINDOW_HALF_MS;
        long w1 = totalDurationMs > 0 ? Math.min(totalDurationMs, rawW1) : rawW1;
        List<AudioSlice> slices = fineSource.sliceRange(w0, w1, FINE_STEP_MS);
        if (slices.size() >= 2) {
            tasks.add(new EdgeTask(segIndex, isStart, internalTransition, centerMs, slices));
        }
    }

    private void refineEdgeBatch(@NonNull List<CandidateSegment> candidates,
                                 @NonNull List<EdgeTask> batch, @NonNull String rubric)
            throws IOException, JSONException, InterruptedException {
        try {
            executeEdgeBatch(candidates, batch, rubric);
        } catch (ProhibitedContentException e) {
            if (batch.size() == 1) {
                Log.d(TAG, "Keeping coarse boundary near " + batch.get(0).centerMs);
                return;
            }
            for (EdgeTask single : batch) {
                refineEdgeBatch(candidates, Collections.singletonList(single), rubric);
            }
        }
    }

    private void executeEdgeBatch(@NonNull List<CandidateSegment> candidates,
                                  @NonNull List<EdgeTask> batch, @NonNull String rubric)
            throws IOException, JSONException, InterruptedException {
        JSONArray input = new JSONArray().put(new JSONObject()
                .put("type", "text")
                .put("text", rubric + "\nFine 3-second audio slices around ad break boundaries:"));
        JSONObject questions = new JSONObject();
        JSONObject criteria = buildChoiceCriteria();
        for (int e = 0; e < batch.size(); e++) {
            EdgeTask task = batch.get(e);
            String kindLabel = task.internalTransition ? "transition" : (task.isStart ? "start" : "end");
            input.put(new JSONObject().put("type", "text")
                    .put("text", "\n--- E" + e + " (" + kindLabel
                            + " near " + formatTimestamp(task.centerMs) + ") ---"));
            appendEdgeSliceQuestions(input, questions, criteria, e, task.slices);
        }
        JSONObject answers = executeDecisions(input, questions);
        for (int e = 0; e < batch.size(); e++) {
            applyRefinedEdge(candidates, batch.get(e), e, answers);
        }
    }

    private static void appendEdgeSliceQuestions(@NonNull JSONArray input, @NonNull JSONObject questions,
                                                 @NonNull JSONObject criteria, int edgeIdx,
                                                 @NonNull List<AudioSlice> slices) throws JSONException {
        for (int j = 0; j < slices.size(); j++) {
            AudioSlice slice = slices.get(j);
            String qid = "e" + edgeIdx + "_" + j;
            input.put(new JSONObject().put("type", "text").put("text", qid + ":"));
            input.put(new JSONObject().put("type", "audio")
                    .put("mime_type", slice.mimeType)
                    .put("data", slice.base64Data));
            questions.put(qid, new JSONObject()
                    .put("type", "choice")
                    .put("instructions", "Classify 3s slice " + qid + ".")
                    .put("criteria", criteria));
        }
    }

    private static void applyRefinedEdge(@NonNull List<CandidateSegment> candidates,
                                         @NonNull EdgeTask task, int edgeIdx,
                                         @NonNull JSONObject answers) {
        List<Long> starts = new ArrayList<>(task.slices.size());
        List<Long> ends = new ArrayList<>(task.slices.size());
        BlockScore[] scores = new BlockScore[task.slices.size()];
        for (int j = 0; j < task.slices.size(); j++) {
            AudioSlice slice = task.slices.get(j);
            starts.add(slice.startMs);
            ends.add(slice.endMs);
            scores[j] = parseAnswerScore(answers.optJSONObject("e" + edgeIdx + "_" + j));
        }
        CandidateSegment cur = candidates.get(task.segmentIndex);
        if (task.internalTransition && task.segmentIndex > 0) {
            CandidateSegment prev = candidates.get(task.segmentIndex - 1);
            long splitMs = fitInternalTypeChangepoint(starts, ends, scores, prev.type);
            prev.endMs = splitMs;
            prev.endRefined = true;
            cur.startMs = splitMs;
            cur.startRefined = true;
        } else if (task.isStart) {
            cur.startMs = fitBoundaryChangepoint(starts, ends, scores, true);
            cur.startRefined = true;
        } else {
            cur.endMs = fitBoundaryChangepoint(starts, ends, scores, false);
            cur.endRefined = true;
        }
    }

    @NonNull
    static List<AdSegment> snapAndFinalizeSegments(@NonNull File file,
                                                   @NonNull List<CandidateSegment> candidates,
                                                   long totalDurationMs) {
        List<AdSegment> raw = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            CandidateSegment cur = candidates.get(i);
            boolean prevAdjacent = i > 0
                    && Math.abs(candidates.get(i - 1).endMs - cur.startMs) <= ADJACENT_TOLERANCE_MS;
            boolean nextAdjacent = i + 1 < candidates.size()
                    && Math.abs(candidates.get(i + 1).startMs - cur.endMs) <= ADJACENT_TOLERANCE_MS;
            long start = cur.startMs;
            long end = cur.endMs;
            if (!prevAdjacent) {
                long cushion = cur.startRefined ? REFINED_CUSHION_MS : COARSE_CUSHION_MS;
                start = AudioPauseSnapper.refineBoundary(file, start, true, totalDurationMs, cushion);
            }
            if (!nextAdjacent) {
                long cushion = cur.endRefined ? REFINED_CUSHION_MS : COARSE_CUSHION_MS;
                end = AudioPauseSnapper.refineBoundary(file, end, false, totalDurationMs, cushion);
            }
            if (totalDurationMs > 0) {
                end = Math.min(totalDurationMs, end);
            }
            if (end - start >= MIN_SEGMENT_MS) {
                raw.add(new AdSegment(start, end, cur.type, true));
            }
        }
        return AdSegment.merge(raw, MERGE_GAP_MS);
    }

    @NonNull
    static boolean[] viterbiSmooth(@NonNull double[] nonContentProbs) {
        int n = nonContentProbs.length;
        if (n == 0) {
            return new boolean[0];
        }
        double logSwitch = Math.log(VITERBI_SWITCH);
        double logStay = Math.log(1.0 - VITERBI_SWITCH);
        double score0 = 0;
        double score1 = 0;
        int[][] back = new int[n][2];
        for (int i = 0; i < n; i++) {
            double p = Math.min(0.999, Math.max(0.001, nonContentProbs[i]));
            double llr = VITERBI_ALPHA * Math.log(p / (1.0 - p)) + VITERBI_BIAS;
            if (i == 0) {
                score0 = 0;
                score1 = llr;
            } else {
                double stay0 = score0 + logStay;
                double from1 = score1 + logSwitch;
                back[i][0] = from1 > stay0 ? 1 : 0;
                double next0 = Math.max(stay0, from1);

                double from0 = score0 + logSwitch;
                double stay1 = score1 + logStay;
                back[i][1] = stay1 >= from0 ? 1 : 0;
                double next1 = Math.max(from0, stay1) + llr;

                score0 = next0;
                score1 = next1;
            }
        }
        int state = score1 > score0 ? 1 : 0;
        boolean[] active = new boolean[n];
        active[n - 1] = state == 1;
        for (int i = n - 1; i >= 1; i--) {
            state = back[i][state];
            active[i - 1] = state == 1;
        }
        return active;
    }

    @NonNull
    static List<CandidateSegment> buildCoarseSegments(@NonNull List<Long> starts,
                                                      @NonNull List<Long> ends,
                                                      @NonNull BlockScore[] scores) {
        double[] nonContent = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            nonContent[i] = scores[i].nonContentProb();
        }
        boolean[] active = viterbiSmooth(nonContent);
        List<CandidateSegment> segments = new ArrayList<>();
        int i = 0;
        while (i < active.length) {
            if (!active[i]) {
                i++;
                continue;
            }
            int runEnd = i;
            while (runEnd < active.length && active[runEnd]) {
                runEnd++;
            }
            appendTypedRunSegments(starts, ends, scores, i, runEnd, segments);
            i = runEnd;
        }
        return segments;
    }

    private static void appendTypedRunSegments(@NonNull List<Long> starts, @NonNull List<Long> ends,
                                               @NonNull BlockScore[] scores, int runStart, int runEnd,
                                               @NonNull List<CandidateSegment> out) {
        int[] types = new int[runEnd - runStart];
        for (int k = runStart; k < runEnd; k++) {
            int lo = Math.max(runStart, k - 1);
            int hi = Math.min(runEnd - 1, k + 1);
            double diffSum = 0;
            for (int m = lo; m <= hi; m++) {
                diffSum += scores[m].selfPromoProb - scores[m].adProb;
            }
            double avgDiff = diffSum / (hi - lo + 1);
            types[k - runStart] = avgDiff > 0.04 ? AdSegment.TYPE_SELF_PROMO : AdSegment.TYPE_AD;
        }
        int subStart = runStart;
        while (subStart < runEnd) {
            int curType = types[subStart - runStart];
            int subEnd = subStart;
            while (subEnd < runEnd && types[subEnd - runStart] == curType) {
                subEnd++;
            }
            out.add(new CandidateSegment(starts.get(subStart), ends.get(subEnd - 1), curType));
            subStart = subEnd;
        }
    }

    static long fitBoundaryChangepoint(@NonNull List<Long> sliceStarts, @NonNull List<Long> sliceEnds,
                                       @NonNull BlockScore[] sliceScores, boolean isStart) {
        int n = sliceScores.length;
        int bestSplit = isStart ? n : 0;
        double bestUtility = -1e9;
        for (int split = 0; split <= n; split++) {
            double utility = 0;
            for (int j = 0; j < n; j++) {
                boolean includeInAd = isStart ? j >= split : j < split;
                if (includeInAd) {
                    double p = sliceScores[j].nonContentProb();
                    utility += p - CONTENT_PROTECTION_PENALTY * (1.0 - p);
                }
            }
            if (utility > bestUtility) {
                bestUtility = utility;
                bestSplit = split;
            }
        }
        if (isStart) {
            return bestSplit < n ? sliceStarts.get(bestSplit) : sliceEnds.get(n - 1);
        }
        return bestSplit > 0 ? sliceEnds.get(bestSplit - 1) : sliceStarts.get(0);
    }

    static long fitInternalTypeChangepoint(@NonNull List<Long> sliceStarts, @NonNull List<Long> sliceEnds,
                                           @NonNull BlockScore[] sliceScores, int leftType) {
        int n = sliceScores.length;
        int bestSplit = n / 2;
        double bestUtility = -1e9;
        for (int split = 1; split < n; split++) {
            double utility = 0;
            for (int j = 0; j < n; j++) {
                int expectedType = j < split ? leftType
                        : (leftType == AdSegment.TYPE_AD ? AdSegment.TYPE_SELF_PROMO : AdSegment.TYPE_AD);
                utility += expectedType == AdSegment.TYPE_SELF_PROMO
                        ? sliceScores[j].selfPromoProb - sliceScores[j].adProb
                        : sliceScores[j].adProb - sliceScores[j].selfPromoProb;
            }
            if (utility > bestUtility) {
                bestUtility = utility;
                bestSplit = split;
            }
        }
        return bestSplit < n ? sliceStarts.get(bestSplit) : sliceEnds.get(n - 1);
    }

    @NonNull
    private JSONObject executeDecisions(@NonNull JSONArray input, @NonNull JSONObject questions)
            throws IOException, JSONException, InterruptedException {
        JSONObject payload = new JSONObject()
                .put("model", MODEL)
                .put("input", input)
                .put("questions", questions);
        Request request = new Request.Builder()
                .url(baseUrl + "/v1beta/decisions")
                .header("x-goog-api-key", apiKey)
                .post(RequestBody.create(payload.toString(), JSON_MEDIA))
                .build();
        int rateLimitAttempts = 0;
        int serverAttempts = 0;
        while (true) {
            try (Response response = client.newCall(request).execute()) {
                String body = response.body() != null ? response.body().string() : "";
                if (response.isSuccessful()) {
                    JSONObject json = new JSONObject(body);
                    JSONObject answers = json.optJSONObject("answers");
                    if (answers == null) {
                        throw new IOException("Missing answers in decisions response");
                    }
                    return answers;
                }
                if (response.code() == 400 && body.contains("prohibited_content")) {
                    throw new ProhibitedContentException("Blocked content: " + body);
                }
                if (response.code() == 429) {
                    boolean hardDailyQuota = body.contains("PerDay");
                    int maxAllowed = hardDailyQuota ? MAX_HARD_QUOTA_RETRIES : MAX_RATE_LIMIT_RETRIES;
                    if (rateLimitAttempts >= maxAllowed) {
                        throw new QuotaExceededException("Request failed: 429 " + body);
                    }
                    rateLimitAttempts++;
                    long waitMs = parseRetryDelayMs(body, baseRetryDelayMs * rateLimitAttempts);
                    Log.d(TAG, "Rate limited (429), sleeping " + waitMs + "ms");
                    Thread.sleep(waitMs);
                    continue;
                }
                if (response.code() >= 500 && serverAttempts < MAX_SERVER_RETRIES) {
                    serverAttempts++;
                    Thread.sleep(Math.max(1L, baseRetryDelayMs / 2) * serverAttempts);
                    continue;
                }
                throw new IOException("Request failed: " + response.code() + " " + body);
            }
        }
    }

    static long parseRetryDelayMs(@NonNull String body, long fallbackMs) {
        if (fallbackMs <= 0) {
            return 0;
        }
        Matcher matcher = RETRY_DELAY_PATTERN.matcher(body);
        if (matcher.find()) {
            try {
                double seconds = Double.parseDouble(matcher.group(1));
                long parsedMs = Math.round(seconds * 1000.0) + 1500L;
                return Math.min(70_000L, Math.max(1_000L, parsedMs));
            } catch (NumberFormatException expected) {
            }
        }
        return fallbackMs;
    }

    @NonNull
    private static BlockScore parseAnswerScore(@Nullable JSONObject answer) {
        if (answer == null) {
            return new BlockScore(0, 0);
        }
        JSONObject probs = answer.optJSONObject("probabilities");
        if (probs != null) {
            return new BlockScore(probs.optDouble("ad", 0.0), probs.optDouble("self_promo", 0.0));
        }
        String choice = answer.optString("choice", "");
        if ("ad".equals(choice)) {
            return new BlockScore(answer.optDouble("confidence", 0.9), 0.0);
        }
        if ("self_promo".equals(choice)) {
            return new BlockScore(0.0, answer.optDouble("confidence", 0.9));
        }
        return new BlockScore(0, 0);
    }

    @NonNull
    private static String buildRubric(@Nullable String episodeTitle, @Nullable String podcastTitle) {
        return "Podcast: \"" + (podcastTitle != null ? podcastTitle : "")
                + "\", Episode: \"" + (episodeTitle != null ? episodeTitle : "") + "\".\n"
                + "Classify each audio segment so a podcast player can skip commercial breaks:\n"
                + "- content: actual episode discussion, journalism, story, intro/outro credits, "
                + "or 'we'll be right back' teasers.\n"
                + "- ad: paid sponsor reads, host-read ads, commercials, movie/podcast trailers, "
                + "network idents, or ad disclaimers. Back-to-back ads in a break are all ad.\n"
                + "- self_promo: the show promoting its own newsletter, app, membership, book, "
                + "merchandise, live tour, or asking listeners to subscribe.\n";
    }

    @NonNull
    private static JSONObject buildChoiceCriteria() throws JSONException {
        return new JSONObject()
                .put("content", "Episode story, host discussion, intro, or teaser")
                .put("ad", "Paid sponsor ad, commercial, external promo, or ident")
                .put("self_promo", "Show's own promo, merch, app, or membership");
    }

    @NonNull
    private static List<AudioSlice> readMp3Slices(@NonNull File file,
                                                  @NonNull List<Mp3FrameSplitter.Chunk> chunks)
            throws IOException {
        List<AudioSlice> slices = new ArrayList<>(chunks.size());
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            for (Mp3FrameSplitter.Chunk chunk : chunks) {
                int len = (int) Math.min(Integer.MAX_VALUE, chunk.byteEnd - chunk.byteStart);
                if (len <= 0) {
                    continue;
                }
                byte[] bytes = new byte[len];
                raf.seek(chunk.byteStart);
                raf.readFully(bytes);
                slices.add(new AudioSlice(chunk.startMs, chunk.endMs, "audio/mpeg",
                        ByteString.of(bytes).base64(), len));
            }
        }
        return slices;
    }

    @NonNull
    private static List<AudioSlice> extractAacSlices(@NonNull AacClipExtractor extractor,
                                                     @NonNull File tempFile, long rangeStartMs,
                                                     long rangeEndMs, long stepMs) throws IOException {
        List<AudioSlice> slices = new ArrayList<>();
        for (long clipStart = rangeStartMs; clipStart < rangeEndMs; clipStart += stepMs) {
            long clipEnd = Math.min(rangeEndMs, clipStart + stepMs);
            if (clipEnd - clipStart < 400L) {
                break;
            }
            long actualStart = extractor.writeClip(clipStart, clipEnd, tempFile);
            byte[] bytes = readAllBytes(tempFile);
            if (bytes.length > 0) {
                slices.add(new AudioSlice(actualStart, clipEnd, "audio/mp4",
                        ByteString.of(bytes).base64(), bytes.length));
            }
        }
        return slices;
    }

    @NonNull
    private static byte[] readAllBytes(@NonNull File file) throws IOException {
        int len = (int) Math.min(Integer.MAX_VALUE, file.length());
        byte[] data = new byte[len];
        try (FileInputStream in = new FileInputStream(file)) {
            int offset = 0;
            while (offset < len) {
                int read = in.read(data, offset, len - offset);
                if (read < 0) {
                    break;
                }
                offset += read;
            }
        }
        return data;
    }

    @NonNull
    private static String formatTimestamp(long ms) {
        long seconds = Math.max(0, ms) / 1000;
        return String.format(Locale.US, "%02d:%02d", seconds / 60, seconds % 60);
    }
}
