package de.danoeh.antennapod.net.download.service.episode;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class Mp3FrameSplitterTest {
    private static final int FRAME_LENGTH = 417;

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    private File createMp3(int numFrames) throws IOException {
        File file = folder.newFile("test.mp3");
        try (FileOutputStream out = new FileOutputStream(file)) {
            byte[] id3 = new byte[10 + 20];
            id3[0] = 'I';
            id3[1] = 'D';
            id3[2] = '3';
            id3[9] = 20;
            out.write(id3);
            byte[] frame = new byte[FRAME_LENGTH];
            frame[0] = (byte) 0xff;
            frame[1] = (byte) 0xfb;
            frame[2] = (byte) 0x90;
            for (int i = 0; i < numFrames; i++) {
                out.write(frame);
            }
        }
        return file;
    }

    @Test
    public void testSplit() throws IOException {
        File file = createMp3(2000);
        List<Mp3FrameSplitter.Chunk> chunks = Mp3FrameSplitter.split(file, 10_000);
        assertEquals(6, chunks.size());
        assertEquals(30, chunks.get(0).byteStart);
        assertEquals(0, (chunks.get(1).byteStart - 30) % FRAME_LENGTH);
        for (int i = 1; i < chunks.size(); i++) {
            assertEquals(chunks.get(i - 1).byteEnd, chunks.get(i).byteStart);
            assertEquals(chunks.get(i - 1).endMs, chunks.get(i).startMs);
        }
        assertEquals(file.length(), chunks.get(chunks.size() - 1).byteEnd);
        assertTrue(Math.abs(chunks.get(chunks.size() - 1).endMs - 52_245) < 10);
    }

    @Test
    public void testNotMp3() throws IOException {
        File file = folder.newFile("test.m4a");
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(new byte[50_000]);
        }
        assertTrue(Mp3FrameSplitter.split(file, 10_000).isEmpty());
    }
}
