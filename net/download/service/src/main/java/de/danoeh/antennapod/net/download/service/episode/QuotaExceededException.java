package de.danoeh.antennapod.net.download.service.episode;

import java.io.IOException;

public class QuotaExceededException extends IOException {
    public QuotaExceededException(String message) {
        super(message);
    }
}
