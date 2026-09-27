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
import okio.BufferedSink;
import okio.Okio;
import okio.Source;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AdSegmentIndexer {
    private static final String TAG = "AdSegmentIndexer";
    private static final String BASE_URL = "https://generativelanguage.googleapis.com";
    private static final String MODEL = "gemini-3.5-flash-lite";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final long TRANSCRIPT_CHUNK_MS = 30 * 60 * 1000L;
    private static final long CLIP_MS = 10 * 60 * 1000L;
    private static final int CLIP_ATTEMPTS = 3;
    private static final int TRANSCRIPTS_PER_CLIP = 2;
    private static final double TRANSCRIPT_TEMPERATURE_STEP = 0.4;
    private static final double WORDS_PER_SECOND = 2.5;
    private static final long START_SILENCE_MS = 3000;
    private static final long MAX_PART_DURATION_MS = 8 * 60 * 60 * 1000L;
    private static final long TIMESTAMP_TOLERANCE_MS = 5000;
    private static final long MERGE_GAP_MS = 5000;
    private static final long CLASSIFY_WINDOW_MS = 20 * 60 * 1000L;
    private static final long CLASSIFY_STEP_MS = 15 * 60 * 1000L;
    private static final int MAX_RETRIES = 5;
    private static final int MAX_QUOTA_RETRIES = 2;
    private static final long RETRY_DELAY_MS = 30000;
    private static final Pattern TRANSCRIPT_LINE = Pattern.compile(
            "^\\s*\\[\\s*(\\d+):(\\d{1,2})(?::(\\d{1,2}))?(?:[.,]\\d+)?\\s*]\\s*(.*)$");

    private final OkHttpClient client;
    private final String apiKey;
    private final File tempDir;

    static class TranscriptLine {
        final long time;
        final String text;

        TranscriptLine(long time, String text) {
            this.time = time;
            this.text = text;
        }
    }

    public AdSegmentIndexer(@NonNull OkHttpClient client, @NonNull String apiKey, @NonNull File tempDir) {
        this.client = client.newBuilder()
                .readTimeout(10, TimeUnit.MINUTES)
                .writeTimeout(10, TimeUnit.MINUTES)
                .build();
        this.apiKey = apiKey;
        this.tempDir = tempDir;
    }

    @NonNull
    public List<AdSegment> index(@NonNull File file, @Nullable String mimeType, long durationMs,
                                 @Nullable String episodeTitle, @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        String uploadMimeType = getUploadMimeType(file, mimeType);
        List<TranscriptLine> lines = new ArrayList<>();
        List<Mp3FrameSplitter.Chunk> clips = "audio/mpeg".equals(uploadMimeType)
                ? Mp3FrameSplitter.split(file, CLIP_MS) : new ArrayList<>();
        if (!clips.isEmpty()) {
            for (Mp3FrameSplitter.Chunk clip : clips) {
                transcribeClip(file, clip.byteStart, clip.byteEnd, "audio/mpeg",
                        clip.startMs, clip.endMs - clip.startMs, lines);
            }
            Log.d(TAG, "Transcript has " + lines.size() + " lines from " + clips.size() + " clips");
            if (lines.isEmpty()) {
                return new ArrayList<>();
            }
            return detectAds(lines, durationMs > 0 ? durationMs : clips.get(clips.size() - 1).endMs,
                    episodeTitle, podcastTitle);
        }
        try (AacClipExtractor aacExtractor = AacClipExtractor.open(file)) {
            if (aacExtractor != null) {
                long totalMs = durationMs > 0 ? durationMs : aacExtractor.getDurationMs();
                File clipFile = File.createTempFile("ad_index", ".m4a", tempDir);
                try {
                    for (long clipStart = 0; clipStart < totalMs; clipStart += CLIP_MS) {
                        long clipEnd = Math.min(totalMs, clipStart + CLIP_MS);
                        long actualStart = aacExtractor.writeClip(clipStart, clipEnd, clipFile);
                        transcribeClip(clipFile, 0, clipFile.length(), "audio/mp4",
                                actualStart, clipEnd - actualStart, lines);
                    }
                } finally {
                    if (clipFile.exists() && !clipFile.delete()) {
                        Log.d(TAG, "Unable to delete temporary clip");
                    }
                }
                Log.d(TAG, "Transcript has " + lines.size() + " lines from AAC clips");
                if (lines.isEmpty()) {
                    return new ArrayList<>();
                }
                return detectAds(lines, totalMs, episodeTitle, podcastTitle);
            }
        }
        int numParts = (int) Math.max(1, (durationMs + MAX_PART_DURATION_MS - 1) / MAX_PART_DURATION_MS);
        long fileSize = file.length();
        for (int i = 0; i < numParts; i++) {
            long byteStart = fileSize * i / numParts;
            long byteEnd = fileSize * (i + 1) / numParts;
            long partOffset = durationMs * i / numParts;
            long partDuration = durationMs * (i + 1) / numParts - partOffset;
            JSONObject uploadedFile = upload(file, byteStart, byteEnd, uploadMimeType);
            try {
                uploadedFile = waitUntilActive(uploadedFile);
                if (partDuration <= 0) {
                    String transcript = transcribe(uploadedFile, -1, -1, 0);
                    parseTranscript(transcript, -1, -1, partOffset, lines);
                    continue;
                }
                for (long chunkStart = 0; chunkStart < partDuration; chunkStart += TRANSCRIPT_CHUNK_MS) {
                    long chunkEnd = Math.min(partDuration, chunkStart + TRANSCRIPT_CHUNK_MS);
                    String transcript = transcribe(uploadedFile, chunkStart, chunkEnd, 0);
                    parseTranscript(transcript, chunkStart, chunkEnd, partOffset, lines);
                }
            } finally {
                deleteUploadedFile(uploadedFile.getString("name"));
            }
        }
        Log.d(TAG, "Transcript has " + lines.size() + " lines");
        if (lines.isEmpty()) {
            return new ArrayList<>();
        }
        return detectAds(lines, durationMs, episodeTitle, podcastTitle);
    }

    private void transcribeClip(File file, long byteStart, long byteEnd, String mimeType, long clipStartMs,
                                long clipLength, List<TranscriptLine> lines)
            throws IOException, JSONException, InterruptedException {
        String best = null;
        String fallback = "";
        double bestMisfit = Double.MAX_VALUE;
        for (int attempt = 0; attempt < CLIP_ATTEMPTS && best == null; attempt++) {
            JSONObject uploadedFile = upload(file, byteStart, byteEnd, mimeType);
            List<String> transcripts;
            try {
                uploadedFile = waitUntilActive(uploadedFile);
                transcripts = transcribeInParallel(uploadedFile, 0.3 * attempt);
            } finally {
                deleteUploadedFile(uploadedFile.getString("name"));
            }
            for (String transcript : transcripts) {
                fallback = transcript;
                if (!hasValidTimestamps(transcript, clipLength)) {
                    continue;
                }
                List<TranscriptLine> candidate = new ArrayList<>();
                parseTranscript(transcript, 0, clipLength, 0, candidate);
                double misfit = getTimingMisfit(candidate, clipLength);
                Log.d(TAG, "Transcript of clip at " + clipStartMs + " has timing misfit " + misfit);
                if (misfit < bestMisfit) {
                    bestMisfit = misfit;
                    best = transcript;
                }
            }
            if (best == null) {
                Log.d(TAG, "Transcripts of clip at " + clipStartMs + " have invalid timestamps, retrying");
            }
        }
        parseTranscript(best != null ? best : fallback, 0, clipLength, clipStartMs, lines);
    }

    /**
     * Transcribes the clip twice in parallel, so that a transcript with wrong timestamps can be discarded.
     */
    private List<String> transcribeInParallel(JSONObject uploadedFile, double extraTemperature)
            throws IOException, JSONException, InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(TRANSCRIPTS_PER_CLIP);
        try {
            List<Future<String>> futures = new ArrayList<>();
            for (int i = 0; i < TRANSCRIPTS_PER_CLIP; i++) {
                double temperature = Math.min(1, TRANSCRIPT_TEMPERATURE_STEP * i + extraTemperature);
                futures.add(executor.submit(() -> transcribe(uploadedFile, -1, -1, temperature)));
            }
            List<String> transcripts = new ArrayList<>();
            for (Future<String> future : futures) {
                try {
                    transcripts.add(future.get());
                } catch (ExecutionException e) {
                    if (e.getCause() instanceof IOException) {
                        throw (IOException) e.getCause();
                    } else if (e.getCause() instanceof JSONException) {
                        throw (JSONException) e.getCause();
                    }
                    throw new IOException(e.getCause());
                }
            }
            return transcripts;
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * Returns the fraction of the clip's duration that the length of the transcribed sentences does not explain,
     * for example silence at the start, gaps between sentences or sentences that follow each other too fast.
     */
    static double getTimingMisfit(List<TranscriptLine> lines, long clipLength) {
        if (lines.isEmpty() || clipLength <= 0) {
            return Double.MAX_VALUE;
        }
        double misfit = Math.max(0, lines.get(0).time - START_SILENCE_MS);
        for (int i = 0; i < lines.size(); i++) {
            double expected = lines.get(i).text.split("\\s+").length * 1000.0 / WORDS_PER_SECOND;
            if (i + 1 < lines.size()) {
                misfit += Math.abs(lines.get(i + 1).time - lines.get(i).time - expected);
            } else {
                misfit += Math.max(0, clipLength - lines.get(i).time - expected - START_SILENCE_MS);
            }
        }
        return misfit / clipLength;
    }

    private static boolean hasValidTimestamps(String transcript, long clipLength) {
        int count = 0;
        long previous = 0;
        for (String line : transcript.split("\n")) {
            Matcher matcher = TRANSCRIPT_LINE.matcher(line);
            if (!matcher.matches()) {
                continue;
            }
            long time = parseSeconds(matcher) * 1000;
            if (time > clipLength + TIMESTAMP_TOLERANCE_MS || time + TIMESTAMP_TOLERANCE_MS < previous) {
                return false;
            }
            previous = time;
            count++;
        }
        return count > 0;
    }

    private static long parseSeconds(Matcher matcher) {
        if (matcher.group(3) != null) {
            return Long.parseLong(matcher.group(1)) * 3600
                    + Long.parseLong(matcher.group(2)) * 60 + Long.parseLong(matcher.group(3));
        }
        return Long.parseLong(matcher.group(1)) * 60 + Long.parseLong(matcher.group(2));
    }

    private static String getUploadMimeType(File file, @Nullable String mimeType) {
        if (mimeType != null && mimeType.startsWith("audio/")) {
            return mimeType;
        }
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".m4a") || name.endsWith(".mp4")) {
            return "audio/mp4";
        } else if (name.endsWith(".ogg") || name.endsWith(".opus")) {
            return "audio/ogg";
        } else if (name.endsWith(".aac")) {
            return "audio/aac";
        } else if (name.endsWith(".wav")) {
            return "audio/wav";
        } else if (name.endsWith(".flac")) {
            return "audio/flac";
        }
        return "audio/mpeg";
    }

    private JSONObject upload(File file, long byteStart, long byteEnd, String mimeType)
            throws IOException, JSONException {
        long length = byteEnd - byteStart;
        JSONObject metadata = new JSONObject().put("file", new JSONObject().put("display_name", file.getName()));
        Request startRequest = new Request.Builder()
                .url(BASE_URL + "/upload/v1beta/files")
                .header("x-goog-api-key", apiKey)
                .header("X-Goog-Upload-Protocol", "resumable")
                .header("X-Goog-Upload-Command", "start")
                .header("X-Goog-Upload-Header-Content-Length", String.valueOf(length))
                .header("X-Goog-Upload-Header-Content-Type", mimeType)
                .post(RequestBody.create(metadata.toString(), JSON))
                .build();
        String uploadUrl;
        try (Response response = client.newCall(startRequest).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Starting upload failed: " + response.code() + " " + readBody(response));
            }
            uploadUrl = response.header("X-Goog-Upload-URL");
        }
        if (uploadUrl == null) {
            throw new IOException("No upload URL received");
        }
        RequestBody body = new RequestBody() {
            @Override
            public MediaType contentType() {
                return MediaType.parse(mimeType);
            }

            @Override
            public long contentLength() {
                return length;
            }

            @Override
            public void writeTo(@NonNull BufferedSink sink) throws IOException {
                try (InputStream in = new FileInputStream(file)) {
                    long skipped = 0;
                    while (skipped < byteStart) {
                        long n = in.skip(byteStart - skipped);
                        if (n <= 0) {
                            throw new IOException("Unable to skip to part start");
                        }
                        skipped += n;
                    }
                    Source source = Okio.source(in);
                    long written = 0;
                    while (written < length) {
                        long n = source.read(sink.getBuffer(), Math.min(8192, length - written));
                        if (n == -1) {
                            throw new IOException("Unexpected end of file");
                        }
                        written += n;
                        sink.emitCompleteSegments();
                    }
                }
            }
        };
        Request uploadRequest = new Request.Builder()
                .url(uploadUrl)
                .header("X-Goog-Upload-Offset", "0")
                .header("X-Goog-Upload-Command", "upload, finalize")
                .post(body)
                .build();
        try (Response response = client.newCall(uploadRequest).execute()) {
            String responseBody = readBody(response);
            if (!response.isSuccessful()) {
                throw new IOException("Upload failed: " + response.code() + " " + responseBody);
            }
            return new JSONObject(responseBody).getJSONObject("file");
        }
    }

    private JSONObject waitUntilActive(JSONObject uploadedFile)
            throws IOException, JSONException, InterruptedException {
        for (int i = 0; i < 300 && "PROCESSING".equals(uploadedFile.optString("state")); i++) {
            Thread.sleep(2000);
            Request request = new Request.Builder()
                    .url(BASE_URL + "/v1beta/" + uploadedFile.getString("name"))
                    .header("x-goog-api-key", apiKey)
                    .build();
            try (Response response = client.newCall(request).execute()) {
                String responseBody = readBody(response);
                if (!response.isSuccessful()) {
                    throw new IOException("Checking file state failed: " + response.code() + " " + responseBody);
                }
                uploadedFile = new JSONObject(responseBody);
            }
        }
        if (!"ACTIVE".equals(uploadedFile.optString("state"))) {
            throw new IOException("Uploaded file is not active: " + uploadedFile.optString("state"));
        }
        return uploadedFile;
    }

    private void deleteUploadedFile(String name) {
        Request request = new Request.Builder()
                .url(BASE_URL + "/v1beta/" + name)
                .header("x-goog-api-key", apiKey)
                .delete()
                .build();
        try (Response response = client.newCall(request).execute()) {
            Log.d(TAG, "Deleted uploaded file: " + response.code());
        } catch (IOException e) {
            Log.d(TAG, "Deleting uploaded file failed: " + e.getMessage());
        }
    }

    private String transcribe(JSONObject uploadedFile, long chunkStart, long chunkEnd, double temperature)
            throws IOException, JSONException, InterruptedException {
        String prompt;
        if (chunkStart >= 0) {
            prompt = "Transcribe the speech in this audio file from " + formatTimestamp(chunkStart)
                    + " to " + formatTimestamp(chunkEnd) + ". "
                    + "Output one line per sentence, each starting with the timestamp in the audio file "
                    + "where the sentence begins, formatted as [MM:SS]. "
                    + "Timestamps must be relative to the start of the full audio file. ";
        } else {
            prompt = "Transcribe the speech in this audio clip. "
                    + "Output one line per sentence, each starting with the timestamp in the clip "
                    + "where the sentence begins, formatted as [MM:SS]. ";
        }
        prompt += "Transcribe all speech verbatim and completely, including advertisements, "
                + "sponsor messages and promotions. Do not skip or summarize anything. "
                + "Output only the transcript lines.";
        JSONObject fileData = new JSONObject()
                .put("mime_type", uploadedFile.getString("mimeType"))
                .put("file_uri", uploadedFile.getString("uri"));
        JSONArray parts = new JSONArray()
                .put(new JSONObject().put("file_data", fileData))
                .put(new JSONObject().put("text", prompt));
        JSONObject request = new JSONObject()
                .put("contents", new JSONArray().put(new JSONObject().put("parts", parts)))
                .put("generationConfig", new JSONObject().put("temperature", temperature));
        return generateContent(request);
    }

    private static void parseTranscript(String transcript, long chunkStart, long chunkEnd, long offset,
                                        List<TranscriptLine> lines) {
        long previousTime = lines.isEmpty() ? 0 : lines.get(lines.size() - 1).time;
        long fallbackTime = Math.max(previousTime, offset + Math.max(0, chunkStart));
        for (String line : transcript.split("\n")) {
            Matcher matcher = TRANSCRIPT_LINE.matcher(line);
            if (!matcher.matches()) {
                continue;
            }
            String text = matcher.group(4).trim();
            if (text.isEmpty()) {
                continue;
            }
            long seconds = parseSeconds(matcher);
            long time = offset + seconds * 1000;
            boolean inChunk = chunkStart < 0 || (seconds * 1000 >= chunkStart - TIMESTAMP_TOLERANCE_MS
                    && seconds * 1000 <= chunkEnd + TIMESTAMP_TOLERANCE_MS);
            if (!inChunk || time < fallbackTime) {
                time = fallbackTime;
            }
            lines.add(new TranscriptLine(time, text));
            fallbackTime = time;
        }
    }

    private List<AdSegment> detectAds(List<TranscriptLine> lines, long durationMs,
                                      @Nullable String episodeTitle, @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        List<AdSegment> segments = new ArrayList<>();
        long lastTime = lines.get(lines.size() - 1).time;
        for (long windowStart = 0; ; windowStart += CLASSIFY_STEP_MS) {
            List<TranscriptLine> window = new ArrayList<>();
            long windowEnd = durationMs;
            for (TranscriptLine line : lines) {
                if (line.time >= windowStart + CLASSIFY_WINDOW_MS) {
                    windowEnd = line.time;
                    break;
                } else if (line.time >= windowStart) {
                    window.add(line);
                }
            }
            if (!window.isEmpty()) {
                segments.addAll(detectAdsInWindow(window, windowEnd, episodeTitle, podcastTitle));
            }
            if (windowStart + CLASSIFY_WINDOW_MS > Math.max(durationMs, lastTime)) {
                break;
            }
        }
        return AdSegment.merge(segments, MERGE_GAP_MS);
    }

    private List<AdSegment> detectAdsInWindow(List<TranscriptLine> lines, long endMs,
                                              @Nullable String episodeTitle, @Nullable String podcastTitle)
            throws IOException, JSONException, InterruptedException {
        StringBuilder transcript = new StringBuilder();
        for (int i = 0; i < lines.size(); i++) {
            transcript.append(i).append(" [").append(formatTimestamp(lines.get(i).time)).append("] ")
                    .append(lines.get(i).text).append('\n');
        }
        String prompt = "You are labeling ad breaks in a podcast so that a player can skip them. "
                + "Below is a numbered, timestamped transcript of the episode \"" + episodeTitle
                + "\" from the podcast \"" + podcastTitle + "\".\n\n"
                + "Mark every segment that is not part of the episode's actual content:\n"
                + "- \"ad\": sponsor reads, host-read ads, pre-roll/mid-roll/post-roll ads, movie trailers, "
                + "promos for other podcasts or products, network idents such as \"This is an iHeart podcast\", "
                + "and legal disclaimers of ads.\n"
                + "- \"self_promo\": the show promoting itself: its own newsletter, membership, Patreon, "
                + "merchandise, YouTube channel, live shows or asking listeners to rate/subscribe.\n\n"
                + "Rules:\n"
                + "- An ad break usually contains several ads back to back. Report the whole break as ONE segment, "
                + "from the first line of the first ad to the last line of the last ad. "
                + "Never split a break because the sponsor changes.\n"
                + "- Do NOT mark the episode's own content, including its cold open, intro, teasers of the episode "
                + "(\"after the break...\", \"coming up...\"), credits or listener mail.\n"
                + "- Ads often start with phrases like \"support for this show comes from\", "
                + "\"this episode is sponsored by\" or \"brought to you by\", "
                + "and end right before the host returns to the topic.\n"
                + "- Ads at the very start or end of the episode are common; include them.\n\n"
                + "Return the first and last line number and the type of each segment.\n\nTranscript:\n"
                + transcript;
        JSONObject itemSchema = new JSONObject()
                .put("type", "OBJECT")
                .put("properties", new JSONObject()
                        .put("first_line", new JSONObject().put("type", "INTEGER"))
                        .put("last_line", new JSONObject().put("type", "INTEGER"))
                        .put("type", new JSONObject().put("type", "STRING")
                                .put("enum", new JSONArray().put("ad").put("self_promo"))))
                .put("required", new JSONArray().put("first_line").put("last_line").put("type"));
        JSONObject generationConfig = new JSONObject()
                .put("temperature", 0)
                .put("responseMimeType", "application/json")
                .put("responseSchema", new JSONObject().put("type", "ARRAY").put("items", itemSchema));
        JSONArray parts = new JSONArray().put(new JSONObject().put("text", prompt));
        JSONObject request = new JSONObject()
                .put("contents", new JSONArray().put(new JSONObject().put("parts", parts)))
                .put("generationConfig", generationConfig);
        JSONArray result = new JSONArray(generateContent(request));

        List<AdSegment> segments = new ArrayList<>();
        for (int i = 0; i < result.length(); i++) {
            JSONObject segment = result.getJSONObject(i);
            int firstLine = segment.getInt("first_line");
            int lastLine = segment.getInt("last_line");
            if (firstLine < 0 || lastLine < firstLine || lastLine >= lines.size()) {
                continue;
            }
            long start = lines.get(firstLine).time;
            long end;
            if (lastLine + 1 < lines.size()) {
                end = lines.get(lastLine + 1).time;
            } else if (endMs > 0) {
                end = endMs;
            } else {
                continue;
            }
            int type = "self_promo".equals(segment.optString("type")) ? AdSegment.TYPE_SELF_PROMO : AdSegment.TYPE_AD;
            segments.add(new AdSegment(start, end, type, true));
        }
        return segments;
    }

    private String generateContent(JSONObject requestJson)
            throws IOException, JSONException, InterruptedException {
        Request request = new Request.Builder()
                .url(BASE_URL + "/v1beta/models/" + MODEL + ":generateContent")
                .header("x-goog-api-key", apiKey)
                .post(RequestBody.create(requestJson.toString(), JSON))
                .build();
        for (int attempt = 0; ; attempt++) {
            try (Response response = client.newCall(request).execute()) {
                String responseBody = readBody(response);
                if (response.code() == 429 && attempt >= MAX_QUOTA_RETRIES) {
                    throw new QuotaExceededException("Request failed: 429 " + responseBody);
                }
                if ((response.code() == 429 || response.code() >= 500) && attempt < MAX_RETRIES) {
                    Log.d(TAG, "Request failed with " + response.code() + ", retrying");
                    Thread.sleep(RETRY_DELAY_MS * (attempt + 1));
                    continue;
                }
                if (!response.isSuccessful()) {
                    throw new IOException("Request failed: " + response.code() + " " + responseBody);
                }
                JSONArray candidates = new JSONObject(responseBody).optJSONArray("candidates");
                if (candidates == null || candidates.length() == 0) {
                    throw new IOException("No candidates in response");
                }
                JSONArray parts = candidates.getJSONObject(0).getJSONObject("content").getJSONArray("parts");
                StringBuilder text = new StringBuilder();
                for (int i = 0; i < parts.length(); i++) {
                    JSONObject part = parts.getJSONObject(i);
                    if (!part.optBoolean("thought", false)) {
                        text.append(part.optString("text", ""));
                    }
                }
                return text.toString();
            }
        }
    }

    private static String readBody(Response response) throws IOException {
        return response.body() != null ? response.body().string() : "";
    }

    private static String formatTimestamp(long ms) {
        long seconds = ms / 1000;
        return String.format(Locale.US, "%02d:%02d", seconds / 60, seconds % 60);
    }
}
