# SPDX-FileCopyrightText: Portions Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoding functions

Functions used to offer various encodings that are needed more than one place in garak
"""

import random
import unicodedata

_ASCII_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def ascii_fold(text: str) -> str:
    """Strip accents and typographic punctuation, e.g. ``"ação – não"`` -> ``"acao - nao"``.

    For encodings that only cover ASCII letters (ROT13, Morse, NATO, ...), which
    would otherwise drop or leave in clear the accented letters of a translated payload.
    Characters with no ASCII form are kept."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_ASCII_PUNCTUATION))
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if not unicodedata.combining(c))
    )


def leetspeak(input: str, transmute_chance: float = 0.6) -> str:
    src_chars = "ieaost"
    dst_chars = "134057"
    processed = input.translate(
        str.maketrans(src_chars.lower() + src_chars.upper(), dst_chars + dst_chars)
    )
    # mix it up
    output = "".join(
        [processed[i] if random.random() > 0.5 else input[i] for i in range(len(input))]
    )
    return output
