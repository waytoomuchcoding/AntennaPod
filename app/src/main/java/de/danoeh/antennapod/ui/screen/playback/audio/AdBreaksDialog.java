package de.danoeh.antennapod.ui.screen.playback.audio;

import android.content.Context;
import android.text.format.DateUtils;
import androidx.annotation.NonNull;
import com.google.android.material.dialog.MaterialAlertDialogBuilder;
import de.danoeh.antennapod.R;
import de.danoeh.antennapod.model.feed.AdSegment;
import de.danoeh.antennapod.model.feed.FeedMedia;
import de.danoeh.antennapod.storage.database.DBWriter;

import java.util.ArrayList;
import java.util.List;

public class AdBreaksDialog {
    private AdBreaksDialog() {
    }

    public static void show(@NonNull Context context, @NonNull FeedMedia media) {
        List<AdSegment> segments = media.getAdSegments();
        MaterialAlertDialogBuilder builder = new MaterialAlertDialogBuilder(context)
                .setTitle(R.string.ad_breaks);
        if (segments == null || segments.isEmpty()) {
            builder.setMessage(segments == null ? R.string.ad_breaks_not_indexed : R.string.ad_breaks_none)
                    .setPositiveButton(android.R.string.ok, null)
                    .show();
            return;
        }
        CharSequence[] labels = new CharSequence[segments.size()];
        boolean[] checked = new boolean[segments.size()];
        for (int i = 0; i < segments.size(); i++) {
            AdSegment segment = segments.get(i);
            labels[i] = context.getString(R.string.ad_break_label,
                    context.getString(segment.getType() == AdSegment.TYPE_SELF_PROMO
                            ? R.string.ad_break_type_self_promo : R.string.ad_break_type_ad),
                    DateUtils.formatElapsedTime(segment.getStart() / 1000),
                    DateUtils.formatElapsedTime(segment.getEnd() / 1000));
            checked[i] = segment.isEnabled();
        }
        builder.setMultiChoiceItems(labels, checked, (dialog, which, isChecked) -> checked[which] = isChecked)
                .setPositiveButton(android.R.string.ok, (dialog, which) -> {
                    List<AdSegment> updated = new ArrayList<>();
                    for (int i = 0; i < segments.size(); i++) {
                        updated.add(segments.get(i).withEnabled(checked[i]));
                    }
                    DBWriter.setAdSegments(media, updated);
                })
                .setNegativeButton(R.string.cancel_label, null)
                .show();
    }
}
