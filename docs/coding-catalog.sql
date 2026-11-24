-- Read-only catalog consumed by the matcher. Load verified mappings and history
-- from a snapshot. Use a separate SQLite copy and test ranges for isolation.
CREATE TABLE coding_context (
    id INTEGER PRIMARY KEY,
    theme TEXT NOT NULL,
    subtheme TEXT NOT NULL,
    category TEXT NOT NULL,
    theme_prefix TEXT NOT NULL,
    subtheme_prefix TEXT NOT NULL,
    -- The shared source code is category_prefix-theme_prefix followed by the sequence.
    category_prefix TEXT NOT NULL,
    range_start INTEGER NOT NULL,
    range_end INTEGER NOT NULL,
    number_width INTEGER NOT NULL,
    UNIQUE(theme, subtheme, category),
    CHECK(range_start >= 0 AND range_end >= range_start AND number_width BETWEEN 1 AND 12)
);
CREATE TABLE coding_subject (
    context_id INTEGER NOT NULL REFERENCES coding_context(id),
    name TEXT NOT NULL,
    -- Design subtheme abbreviation, used as the first operation-code segment.
    prefix TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY(context_id, name)
);
CREATE TABLE coding_history (
    context_id INTEGER NOT NULL REFERENCES coding_context(id),
    image_sha256 TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    abbreviation TEXT NOT NULL,
    -- Keep the full issued source code, including every prefix and separator.
    source_code TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE INDEX coding_history_image ON coding_history(context_id, image_sha256);
