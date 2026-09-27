package de.danoeh.antennapod.net.download.service.episode;

import android.Manifest;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import androidx.core.content.ContextCompat;
import androidx.work.BackoffPolicy;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.ExistingWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;
import de.danoeh.antennapod.event.FeedItemEvent;
import de.danoeh.antennapod.model.feed.AdSegment;
import de.danoeh.antennapod.model.feed.FeedItem;
import de.danoeh.antennapod.model.feed.FeedMedia;
import de.danoeh.antennapod.net.common.AntennapodHttpClient;
import de.danoeh.antennapod.storage.database.DBReader;
import de.danoeh.antennapod.storage.database.DBWriter;
import de.danoeh.antennapod.net.download.service.R;
import de.danoeh.antennapod.storage.preferences.UserPreferences;
import de.danoeh.antennapod.ui.appstartintent.MainActivityStarter;
import de.danoeh.antennapod.ui.notifications.NotificationUtils;
import org.greenrobot.eventbus.EventBus;
import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

public class AdSegmentIndexWorker extends Worker {
    private static final String TAG = "AdSegmentIndexWorker";
    private static final String WORK_DATA_MEDIA_ID = "media_id";
    private static final String WORK_ID_PREFIX = "AdSegmentIndex_";
    private static final int MAX_ATTEMPTS = 3;
    private static final long QUOTA_NOTIFICATION_INTERVAL_MS = 6 * 60 * 60 * 1000L;
    private static long lastQuotaNotification = 0;
    private static final Set<Long> PENDING = Collections.synchronizedSet(new HashSet<>());

    public AdSegmentIndexWorker(@NonNull Context context, @NonNull WorkerParameters params) {
        super(context, params);
    }

    public static void enqueue(@NonNull Context context, @NonNull FeedMedia media) {
        if (!isEnabledFor(media)) {
            return;
        }
        OneTimeWorkRequest request = new OneTimeWorkRequest.Builder(AdSegmentIndexWorker.class)
                .setConstraints(new Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 5, TimeUnit.MINUTES)
                .setInputData(new Data.Builder().putLong(WORK_DATA_MEDIA_ID, media.getId()).build())
                .build();
        WorkManager.getInstance(context).enqueueUniqueWork(WORK_ID_PREFIX + media.getId(),
                ExistingWorkPolicy.REPLACE, request);
        PENDING.add(media.getId());
        if (media.getItem() != null) {
            EventBus.getDefault().post(new FeedItemEvent(Collections.singletonList(media.getItem()), false));
        }
    }

    public static boolean isIndexing(long mediaId) {
        return PENDING.contains(mediaId);
    }

    private static boolean isEnabledFor(@NonNull FeedMedia media) {
        if (!UserPreferences.isAdSkippingEnabled() || UserPreferences.getGeminiApiKey().isEmpty()) {
            return false;
        }
        FeedItem item = media.getItem();
        return item == null || item.getFeed() == null || item.getFeed().getPreferences() == null
                || item.getFeed().getPreferences().isAdSkippingEnabled();
    }

    private void showQuotaNotification() {
        synchronized (AdSegmentIndexWorker.class) {
            if (System.currentTimeMillis() - lastQuotaNotification < QUOTA_NOTIFICATION_INTERVAL_MS) {
                return;
            }
            lastQuotaNotification = System.currentTimeMillis();
        }
        Context context = getApplicationContext();
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            return;
        }
        PendingIntent intent = PendingIntent.getActivity(context, R.id.pending_intent_ad_detection_quota,
                new MainActivityStarter(context).getIntent(),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        NotificationCompat.Builder builder = new NotificationCompat.Builder(context,
                NotificationUtils.CHANNEL_ID_DOWNLOAD_ERROR)
                .setContentTitle(context.getString(R.string.ad_detection_quota_title))
                .setContentText(context.getString(R.string.ad_detection_quota_message))
                .setStyle(new NotificationCompat.BigTextStyle()
                        .bigText(context.getString(R.string.ad_detection_quota_message)))
                .setSmallIcon(R.drawable.ic_notification_sync_error)
                .setContentIntent(intent)
                .setAutoCancel(true)
                .setVisibility(NotificationCompat.VISIBILITY_PUBLIC);
        NotificationManager nm = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        nm.notify(R.id.notification_ad_detection_quota, builder.build());
    }

    @Override
    @NonNull
    public Result doWork() {
        long mediaId = getInputData().getLong(WORK_DATA_MEDIA_ID, 0);
        PENDING.add(mediaId);
        Result result = Result.failure();
        try {
            result = index(mediaId);
            return result;
        } finally {
            if (!Result.retry().equals(result)) {
                PENDING.remove(mediaId);
                FeedMedia media = DBReader.getFeedMedia(mediaId);
                if (media != null && media.getItem() != null) {
                    EventBus.getDefault().post(new FeedItemEvent(Collections.singletonList(media.getItem()), false));
                }
            }
        }
    }

    @NonNull
    private Result index(long mediaId) {
        String apiKey = UserPreferences.getGeminiApiKey();
        if (!UserPreferences.isAdSkippingEnabled() || apiKey.isEmpty()) {
            return Result.success();
        }
        FeedMedia media = DBReader.getFeedMedia(mediaId);
        if (media == null || !media.localFileAvailable() || !isEnabledFor(media)) {
            return Result.failure();
        }
        List<AdSegment> segments;
        try {
            segments = new AdSegmentIndexer(AntennapodHttpClient.getHttpClient(), apiKey,
                    getApplicationContext().getCacheDir())
                    .index(new File(media.getLocalFileUrl()), media.getMimeType(), media.getDuration(),
                            media.getEpisodeTitle(), media.getFeedTitle());
            FeedMedia currentMedia = DBReader.getFeedMedia(media.getId());
            if (currentMedia == null || !currentMedia.localFileAvailable()) {
                return Result.failure();
            }
            DBWriter.setAdSegments(currentMedia, segments).get();
        } catch (QuotaExceededException e) {
            Log.e(TAG, "Gemini quota exceeded, trying again later", e);
            showQuotaNotification();
            return Result.retry();
        } catch (IOException | JSONException | ExecutionException e) {
            Log.e(TAG, "Indexing ads failed", e);
            return getRunAttemptCount() + 1 < MAX_ATTEMPTS ? Result.retry() : Result.failure();
        } catch (InterruptedException e) {
            return Result.retry();
        }
        Log.d(TAG, "Found " + segments.size() + " ad segments in " + media.getEpisodeTitle());
        return Result.success();
    }
}
