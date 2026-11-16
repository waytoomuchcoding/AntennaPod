package de.danoeh.antennapod.net.download.service.episode;

import de.danoeh.antennapod.model.feed.AdSegment;
import okhttp3.Interceptor;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;
import okio.Buffer;
import org.json.JSONException;
import org.json.JSONObject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

@RunWith(RobolectricTestRunner.class)
public class DecisionAdSegmentIndexerHttpTest {
    private static final MediaType JSON_TYPE = MediaType.get("application/json; charset=utf-8");
    private static final int MP3_FRAME_LENGTH = 417;

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private File createMp3Seconds(int approxSeconds) throws IOException {
        int numFrames = Math.max(120, (int) Math.round(approxSeconds * 1000.0 / 26.122));
        File file = folder.newFile("episode_" + approxSeconds + ".mp3");
        try (FileOutputStream out = new FileOutputStream(file)) {
            byte[] frame = new byte[MP3_FRAME_LENGTH];
            frame[0] = (byte) 0xff;
            frame[1] = (byte) 0xfb;
            frame[2] = (byte) 0x90;
            for (int i = 0; i < numFrames; i++) {
                out.write(frame);
            }
        }
        return file;
    }

    private static String requestBodyString(Request request) throws IOException {
        Buffer buffer = new Buffer();
        if (request.body() != null) {
            request.body().writeTo(buffer);
        }
        return buffer.readUtf8();
    }

    private static Response jsonResponse(Request request, int code, String body) {
        return new Response.Builder()
                .request(request)
                .protocol(Protocol.HTTP_1_1)
                .code(code)
                .message(code == 200 ? "OK" : "Error")
                .body(ResponseBody.create(body, JSON_TYPE))
                .build();
    }

    private static JSONObject choiceAnswer(double adProb, double selfPromoProb) throws JSONException {
        double contentProb = Math.max(0.0, 1.0 - adProb - selfPromoProb);
        String winner = adProb >= selfPromoProb && adProb >= contentProb ? "ad"
                : (selfPromoProb >= contentProb ? "self_promo" : "content");
        return new JSONObject()
                .put("type", "choice")
                .put("choice", winner)
                .put("probabilities", new JSONObject()
                        .put("content", contentProb)
                        .put("ad", adProb)
                        .put("self_promo", selfPromoProb));
    }

    @Test
    public void testEndToEndMp3TwoPassRefinementAndSelfPromo() throws Exception {
        File mp3 = createMp3Seconds(180);
        AtomicInteger callCount = new AtomicInteger();
        Interceptor interceptor = chain -> {
            int call = callCount.getAndIncrement();
            Request req = chain.request();
            assertEquals("/v1beta/decisions", req.url().encodedPath());
            try {
                JSONObject payload = new JSONObject(requestBodyString(req));
                assertEquals("gemini-decision-1", payload.getString("model"));
                JSONObject questions = payload.getJSONObject("questions");
                JSONObject answers = new JSONObject();
                Iterator<String> keys = questions.keys();
                while (keys.hasNext()) {
                    String qid = keys.next();
                    if (call == 0) {
                        if ("b2".equals(qid) || "b3".equals(qid)) {
                            answers.put(qid, choiceAnswer(0.93, 0.02));
                        } else if ("b6".equals(qid)) {
                            answers.put(qid, choiceAnswer(0.05, 0.88));
                        } else {
                            answers.put(qid, choiceAnswer(0.01, 0.01));
                        }
                    } else {
                        int sliceIdx = Integer.parseInt(qid.substring(qid.indexOf('_') + 1));
                        if (qid.startsWith("e0_")) {
                            answers.put(qid, choiceAnswer(sliceIdx >= 8 ? 0.95 : 0.02, 0.0));
                        } else if (qid.startsWith("e1_")) {
                            answers.put(qid, choiceAnswer(sliceIdx < 6 ? 0.95 : 0.02, 0.0));
                        } else if (qid.startsWith("e2_")) {
                            answers.put(qid, choiceAnswer(0.0, sliceIdx >= 7 ? 0.92 : 0.02));
                        } else {
                            answers.put(qid, choiceAnswer(0.0, sliceIdx < 8 ? 0.92 : 0.02));
                        }
                    }
                }
                return jsonResponse(req, 200, new JSONObject().put("answers", answers).toString());
            } catch (JSONException e) {
                throw new IOException(e);
            }
        };
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(interceptor).build();
        DecisionAdSegmentIndexer indexer = new DecisionAdSegmentIndexer(
                client, "test-key", folder.getRoot(), "https://example.invalid", 0L);
        List<Integer> progressUpdates = new ArrayList<>();
        indexer.setProgressListener(progressUpdates::add);

        List<AdSegment> segments = indexer.index(mp3, "audio/mpeg", 180_000L, "Ep 1", "Show");
        assertEquals(2, callCount.get());
        assertEquals(2, segments.size());
        assertEquals(AdSegment.TYPE_AD, segments.get(0).getType());
        assertTrue(segments.get(0).getStart() >= 42_000L && segments.get(0).getStart() <= 46_000L);
        assertTrue(segments.get(0).getEnd() >= 74_000L && segments.get(0).getEnd() <= 78_000L);
        assertEquals(AdSegment.TYPE_SELF_PROMO, segments.get(1).getType());
        assertEquals(Integer.valueOf(100), progressUpdates.get(progressUpdates.size() - 1));
    }

    @Test
    public void testProhibitedContentBisectionRecoversAdBreak() throws Exception {
        File mp3 = createMp3Seconds(200);
        Interceptor interceptor = chain -> {
            Request req = chain.request();
            try {
                JSONObject payload = new JSONObject(requestBodyString(req));
                JSONObject questions = payload.getJSONObject("questions");
                if (questions.has("b1")) {
                    return jsonResponse(req, 400, "{\"error\":{\"code\":\"prohibited_content\"}}");
                }
                JSONObject answers = new JSONObject();
                Iterator<String> keys = questions.keys();
                while (keys.hasNext()) {
                    String qid = keys.next();
                    if ("b7".equals(qid) || "b8".equals(qid)) {
                        answers.put(qid, choiceAnswer(0.95, 0.0));
                    } else if (qid.startsWith("e0_")) {
                        int idx = Integer.parseInt(qid.substring(3));
                        answers.put(qid, choiceAnswer(idx >= 7 ? 0.94 : 0.02, 0.0));
                    } else if (qid.startsWith("e1_")) {
                        int idx = Integer.parseInt(qid.substring(3));
                        answers.put(qid, choiceAnswer(idx < 7 ? 0.94 : 0.02, 0.0));
                    } else {
                        answers.put(qid, choiceAnswer(0.01, 0.01));
                    }
                }
                return jsonResponse(req, 200, new JSONObject().put("answers", answers).toString());
            } catch (JSONException e) {
                throw new IOException(e);
            }
        };
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(interceptor).build();
        DecisionAdSegmentIndexer indexer = new DecisionAdSegmentIndexer(
                client, "test-key", folder.getRoot(), "https://example.invalid", 0L);

        List<AdSegment> segments = indexer.index(mp3, "audio/mpeg", 200_000L, "Crime Ep", "Crime Show");
        assertEquals(1, segments.size());
        assertTrue(segments.get(0).getStart() >= 135_000L && segments.get(0).getStart() <= 145_000L);
    }

    @Test
    public void testRateLimitRetryThenQuotaExceeded() throws Exception {
        File mp3 = createMp3Seconds(60);
        AtomicInteger attempts = new AtomicInteger();
        Interceptor transientInterceptor = chain -> {
            Request req = chain.request();
            if (attempts.getAndIncrement() == 0) {
                return jsonResponse(req, 429, "{\"error\":{\"message\":\"Please retry in 1.2s\"}}");
            }
            try {
                JSONObject questions = new JSONObject(requestBodyString(req)).getJSONObject("questions");
                JSONObject answers = new JSONObject();
                Iterator<String> keys = questions.keys();
                while (keys.hasNext()) {
                    answers.put(keys.next(), choiceAnswer(0.01, 0.01));
                }
                return jsonResponse(req, 200, new JSONObject().put("answers", answers).toString());
            } catch (JSONException e) {
                throw new IOException(e);
            }
        };
        OkHttpClient okClient = new OkHttpClient.Builder().addInterceptor(transientInterceptor).build();
        DecisionAdSegmentIndexer okIndexer = new DecisionAdSegmentIndexer(
                okClient, "test-key", folder.getRoot(), "https://example.invalid", 0L);
        assertTrue(okIndexer.index(mp3, "audio/mpeg", 60_000L, "Ep", "Show").isEmpty());
        assertEquals(2, attempts.get());

        OkHttpClient quotaClient = new OkHttpClient.Builder()
                .addInterceptor(chain -> jsonResponse(chain.request(), 429,
                        "{\"error\":{\"message\":\"GenerateRequestsPerDay exceeded\"}}"))
                .build();
        DecisionAdSegmentIndexer quotaIndexer = new DecisionAdSegmentIndexer(
                quotaClient, "test-key", folder.getRoot(), "https://example.invalid", 0L);
        assertThrows(QuotaExceededException.class,
                () -> quotaIndexer.index(mp3, "audio/mpeg", 60_000L, "Ep", "Show"));
    }
}
