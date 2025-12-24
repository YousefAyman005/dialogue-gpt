import os
from convokit import Corpus, download

# output path
out_dir = "data/movies"
out_file = os.path.join(out_dir, "input.txt")

os.makedirs(out_dir, exist_ok=True)

print("Loading Cornell Movie-Dialogs Corpus...")
corpus = Corpus(filename=download("movie-corpus"))

num_written = 0

with open(out_file, "w", encoding="utf-8") as f:
    for utt in corpus.iter_utterances():
        text = utt.text
        if not text:
            continue

        # get character name if available
        speaker = utt.speaker.meta.get("character_name") or utt.speaker.id

        # basic cleaning
        speaker = speaker.replace("\n", " ").strip()
        text = text.replace("\n", " ").strip()

        # skip very short noise
        if len(text) < 2:
            continue

        f.write(f"{speaker}: {text}\n")
        num_written += 1

print(f"Done. Wrote {num_written} dialogue lines to {out_file}")
