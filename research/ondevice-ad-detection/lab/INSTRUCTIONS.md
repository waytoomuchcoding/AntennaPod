# Labelling podcast transcripts for advertisements

You label which lines of a podcast transcript are advertisements. The labels train an on-device ad skipper, so
precise, consistent boundaries matter more than speed.

## Input
`lab/in/<episode>.txt`: a header (podcast name, episode title, line count), then one line per transcript line:
`L<number> [mm:ss] text`. The text is automatic speech recognition, so expect misspellings
(e.g. "squarespace dot com slash" or "Better help"). Lines are ~5-10 s of speech.

## What to mark
- **ad**: anything that sells or promotes something other than this episode's own content:
  - sponsor reads and commercials, whether read by the host ("this episode is brought to you by...") or produced
    spots with other voices; dynamically inserted ads at the start, middle or end
  - promos/trailers for OTHER podcasts (including other shows from the same network), TV shows, movies, books,
    events, apps; these can sound like story content (a movie trailer in a crime show is still an ad)
  - skit-style or conversational ads that only reveal the brand later: the whole read is ad, from its first line
- **self_promo**: the show promoting ITSELF: subscribe/rate/review asks, its own Patreon/merch/newsletter/live tour,
  "follow us on Instagram", end credits and production credits
- everything else is content (do not list it): the story, interviews, discussion, news, the show's own intro and
  outro music/greetings, and the hosts merely mentioning a company or product in conversation

## Boundaries
- `from` = the first line where the ad starts, `to` = the last line of the ad (inclusive). Breaks often contain
  several ads back to back: one segment for the whole run of consecutive ad lines is fine.
- A line that is mostly ad belongs to the ad; a line that is mostly content does not.
- Host transitions INTO a break ("we'll be right back after this", "but first, a word from our sponsors") belong
  to the ad. "And we're back" belongs to the content.

## Output
Write `lab/out/<episode>.json` (exactly this format; an episode with no ads gets an empty list):
```json
{"labeller": "haiku", "segments": [{"from": 0, "to": 9, "type": "ad", "what": "YouTube Premium; Link"},
                                   {"from": 412, "to": 418, "type": "self_promo", "what": "credits"}]}
```
`what` is a few words naming the advertiser(s) or the promo. Read the WHOLE transcript before writing: ads appear
anywhere, including the very first and very last lines.

## Rules
- Only read files under `lab/in/` and write files under `lab/out/`. Do not run commands or modify anything else.
- Reply with one short line per episode: `<episode>: <n> segments` (no other commentary).
