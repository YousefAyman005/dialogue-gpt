import ast
import os
import re

import kagglehub

LEGACY_DELIM = " +++$+++ "
END_TOKEN = "<|endoftext|>"

# output path
out_dir = "data/movies"
out_file = os.path.join(out_dir, "input.txt")


def clean_text(value):
    if value is None:
        return ""
    return " ".join(value.replace("\n", " ").split()).strip()


def find_dataset_file(root, *filenames):
    for dirpath, _, dir_files in os.walk(root):
        for filename in filenames:
            if filename in dir_files:
                return os.path.join(dirpath, filename)
    expected = ", ".join(filenames)
    raise FileNotFoundError(f"Could not find {expected} under {root}")


def parse_list_field(raw):
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    if raw.startswith("[") and raw.endswith("]"):
        items = re.findall(r"'([^']*)'", raw)
        if items:
            return [clean_text(item) for item in items if clean_text(item)]
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    return [clean_text(item) for item in items if clean_text(item)]


def split_fields(line, delim):
    return line.rstrip("\n").split(delim)


def detect_delim(path):
    if path.endswith(".tsv"):
        return "\t"
    return LEGACY_DELIM


def load_movie_metadata(path):
    delim = detect_delim(path)
    movies = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = split_fields(line, delim)
            if len(parts) < 6:
                continue
            movie_id, title, year, _, _, genres_raw = parts[:6]
            title = clean_text(title)
            year = clean_text(year) or "Unknown"
            movies[movie_id] = {
                "title": title,
                "year": year,
                "genres": parse_list_field(genres_raw),
            }
    return movies


def load_characters(path):
    delim = detect_delim(path)
    characters = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = split_fields(line, delim)
            if len(parts) < 6:
                continue
            character_id, character_name, _, _, _, _ = parts[:6]
            name = clean_text(character_name)
            if name:
                characters[character_id] = name
    return characters


def load_lines(path):
    delim = detect_delim(path)
    lines = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = split_fields(line, delim)
            if len(parts) < 5:
                continue
            line_id, character_id, movie_id, character_name, text = parts[:5]
            lines[line_id] = {
                "character_id": character_id,
                "movie_id": movie_id,
                "character_name": clean_text(character_name),
                "text": clean_text(text),
            }
    return lines


def load_conversations(path):
    delim = detect_delim(path)
    conversations = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = split_fields(line, delim)
            if len(parts) < 4:
                continue
            _, _, movie_id, line_ids_raw = parts[:4]
            line_ids = parse_list_field(line_ids_raw)
            if not line_ids:
                continue
            conversations.append({"movie_id": movie_id, "line_ids": line_ids})
    return conversations


def build_dataset():
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading Cornell Movie-Dialogs Corpus via kagglehub...")
    dataset_path = kagglehub.dataset_download("Cornell-University/movie-dialog-corpus")
    print(f"Dataset path: {dataset_path}")

    movie_path = find_dataset_file(
        dataset_path, "movie_titles_metadata.txt", "movie_titles_metadata.tsv"
    )
    char_path = find_dataset_file(
        dataset_path, "movie_characters_metadata.txt", "movie_characters_metadata.tsv"
    )
    lines_path = find_dataset_file(
        dataset_path, "movie_lines.txt", "movie_lines.tsv"
    )
    conv_path = find_dataset_file(
        dataset_path, "movie_conversations.txt", "movie_conversations.tsv"
    )

    movies = load_movie_metadata(movie_path)
    characters = load_characters(char_path)
    lines = load_lines(lines_path)
    conversations = load_conversations(conv_path)

    num_convos = 0
    num_lines = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for convo in conversations:
            movie = movies.get(convo["movie_id"])
            if not movie:
                continue

            convo_lines = []
            speakers = []
            speaker_set = set()

            for line_id in convo["line_ids"]:
                line = lines.get(line_id)
                if not line:
                    continue
                speaker = characters.get(line["character_id"]) or line["character_name"]
                speaker = clean_text(speaker)
                text = clean_text(line["text"])
                if not speaker or not text:
                    continue
                if speaker not in speaker_set:
                    speakers.append(speaker)
                    speaker_set.add(speaker)
                convo_lines.append(f"{speaker}: {text}")

            if len(convo_lines) < 2:
                continue

            genres = movie["genres"] or ["Unknown"]

            header = [
                f"GENRES={'|'.join(genres)}",
                f"CHARACTERS={'|'.join(speakers)}",
            ]

            f.write("\n".join(header) + "\n")
            f.write("\n".join(convo_lines) + "\n")
            f.write(END_TOKEN + "\n")

            num_convos += 1
            num_lines += len(convo_lines)

    print(f"Done. Wrote {num_convos} conversations and {num_lines} lines to {out_file}")


if __name__ == "__main__":
    build_dataset()
