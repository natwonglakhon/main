# All You Need to Know About Taylor Swift Lyrics

*June 2026*

As someone who grew up and learn English with her musics and to honour her [songwriters hall of fame induction](https://www.songhall.org/profiles/ts), it would be fun to learn more about her musics. Here, I analysed 251 Taylor Swift songs across 12 studio albums, movie soundtracks, and vault tracks. How many words does she use? Which albums have the richest vocabulary? What are her most signature words? And which songs are lyrically most similar to each other?

All lyrics were retrieved from the [Genius API](https://genius.com/api-clients) using `lyricsgenius` and `BeautifulSoup`, then cleaned and analysed in Python with `pandas`, `scikit-learn`, and `seaborn`. The full code is available on [GitHub](https://github.com/natwonglakhon/Taylor_Swift_Lyrics).

## For non-technical people

If you want to play with data, the cvs file for all lyrics data are [here](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/main/taylor_swift_genius_data.csv).


---

## The Data

The dataset covers 251 songs across 12 albums (Taylor's Version releases where available), plus non-album tracks including movie soundtracks and vault releases. Yes, *"I Knew It, I Knew You"* is included. This analysis was conducted in mid-June 2026.

Albums included:
- Taylor Swift (2006)
- Fearless Taylor's Version (2008, 2020)
- Speak Now Taylor's Version (2010, 2023)
- Red Taylor's Version (2012, 2021)
- 1989 Taylor's Version (2014, 2023)
- Reputation (2017)
- Lover (2019)
- Folklore Deluxe (2020)
- Evermore Deluxe (2020)
- Midnights 3am Edition (2022)
- The Tortured Poets Department: The Anthology (2024)
- The Life of a Showgirl (2025)

---

## I. Analysing Lyrics by Album

### Word Count per Album

The first thing I looked at was how many words each album contains in total, and how many of those words are unique — a proxy for vocabulary richness.

**Reputation** has the highest mean word count per track, making it her most lyrically dense album on average. On the other end, **The Life of a Showgirl** has the highest mean *unique* word count per track, suggesting more varied vocabulary relative to its size. (Shocking..)

![Album word count — unique vs non-unique words, sorted by words per track](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/Total_words_per_album.png?raw=true)


When looking at the ratio of unique to total words across the entire album, **The Tortured Poets Department: The Anthology** leads with a 48% unique word ratio (nearly half of all words used are distinct). This is remarkable for a double album of that length.

### Word Count Over Time

Tracing word count chronologically from 2006 to 2025, there is a general growth in lyrical output, with a regression in vocabulary uniqueness between 2006 and 2017. After the Folklore era (2020), the trend in unique vocabulary turned positive again.

![Total and unique word counts by album over time](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/Word_per_track_over_time.png?raw=true)

### Signature Words per Album (TF-IDF)

To find the most characteristic words for each album (words that are common within that album but rare across others), I used TF-IDF (Term Frequency–Inverse Document Frequency). See more in my Jupyter notebook for definition and explanation.

Some standout results:

- **Lover era**: *"Daylight"* is an overwhelmingly obvious signature word, driven by the song of the same name.
- **1989 era**: *"Wonderland"* stands out as a distinctive term.
- **Folklore/Evermore**: longer, more literary words dominate the signatures, consistent with the album's aesthetic.
- **TTPD**: highly specific vocabulary that rarely appears in earlier albums.

![Top 3 TF-IDF signature words per album, sorted by year](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/Signature_words_per_album.png?raw=true)

---

## II. Analysing Words by Song

### Longest and Shortest Songs by Word Count

The song with the most words is, unsurprisingly, **"All Too Well (10-Minute Version)"** from Red Taylor's Version.

The shortest by word count is **"It's Nice to Have a Friend"** from Lover, a 2:30 minute ambient-pop track that is more texture than narrative.

![Top 10 songs by word count](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/top_songs_by_words.png?raw=true)

### Most Unique Songs

The most interesting finding from the song-level analysis is vocabulary uniqueness. **"The Manuscript"** from The Tortured Poets Department is the most unique song in her discography, with 68% of its words being distinct (meaning she barely repeats herself throughout the entire song).

Among the top 10 most unique songs, **12 entries are from The Tortured Poets Department: The Anthology**. This is quite impressive and it suggests that the double album represents a deliberate expansion of her lyrical range.

On the opposite end, **"I Wish You Would"** from 1989 has the lowest unique word ratio, driven by its highly repetitive hook structure.

![Top 10 songs by unique word ratio](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/top_unique_songs_by_ratio.png?raw=true)

![Least 10 songs by unique word ratio](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/430250fb615ad731350fb1527590f50d1f9f5b57/images/least_unique_songs_by_ratio.png?raw=true)

### Word Count Distribution

The distribution of word counts per song is right-skewed where most songs sit in a moderate range, with a small number of outliers (like "All Too Well 10-min version") pulling the tail upward. The histogram reveals a handful of songs well above the interquartile range.

![Word count distribution per song — histogram](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/bfcf29fe7e57218fb3bcc0726c860790b7693df5/images/word_histograms.png?raw=true)

---

## III. Analysing All Words Across 251 Songs

### The Longest Words

Filtering for words of length $\ge 7$ and removing duplicates per song, the longest word in her entire discography is:

> **"Miscommunications"** — 17 letters

It appears in two songs: *"Story of Us"* from Speak Now and *"Question...?"* from Midnights.

The second longest is **"Sanctimoniously"** (15 letters), from *"But Daddy I Love Him"* on The Tortured Poets Department.

![Top 20 longest words across all songs](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/824774910d19c4fb12c42e2059f67ee7ccbc01c7/images/top_longest_words.png?raw=true)

### Most Used Words (Length ≥ 7)

The most frequently used *noun* word with seven or more letters across all 251 songs is **"Forever"**, appearing 10 times in *"Elizabeth Taylor"* from The Life of a Showgirl alone.

**"Daylight"** is the most repetitive long word within a single song, appearing 38 times in *"Daylight"* from Lover.

**"Trouble"** comes second, appearing 24 times in *"I Knew You Were Trouble"* from Red.


### Most Common Consecutive Word Pairs (Bigrams)

Looking at the most frequent two-word combinations across all lyrics:

1. **"And I"** — the most common bigram by a wide margin
2. **"In the"** — a close second


### Song Similarity (Cosine Similarity)

Using *TF-IDF vectors* and *cosine similarity*, I computed a pairwise similarity score between every song pair. The results:

- **Most similar**: *All Too Well (10-Minute Version)* and *All Too Well*, unsurprisingly of course, since one is an extension of the other, sharing the bulk of their lyrics
- **Second most similar**: *Dorothea* and *Safe & Sound* with 55% similarity, which is a genuinely interesting pairing across very different eras (Evermore and The Hunger Games soundtrack respectively)

![Top 20 most lyrically similar song pairs](https://github.com/natwonglakhon/Taylor_Swift_Lyrics/blob/430250fb615ad731350fb1527590f50d1f9f5b57/images/Top_similar_songs.png?raw=true)

---

## IV. Conclusion

Across 251 songs and 20 years of releases, some patterns emerge clearly:

- Vocabulary richness has grown over time, particularly after the Folklore era
- The Tortured Poets Department represents a high-water mark for lyrical variety and unique vocabulary
- Reputation is her most word-dense album per track on average
- "Forever", "Daylight", and "Trouble" are among her most repeated long-form words
- Song similarity analysis surfaces some genuinely surprising pairings beyond the obvious ones

### Limitations

A few caveats worth noting. First, words with different tenses are treated as distinct, e.g., "remember" and "remembered" count separately, which inflates vocabulary counts slightly. Second, songs commonly associated with Taylor Swift that she did not perform (such as *"This Is What You Came For"*) are not included. Third, because Taylor's Version releases are used where available, some original lyrics differ, e.g., the original *"Better Than Revenge"* contains lyrics not present in this dataset, which was a deliberate choice.

---

*Analysis conducted in Python using `pandas`, `scikit-learn`, `nltk`, `seaborn`, and `matplotlib`. Lyrics retrieved via the Genius API using `lyricsgenius`.*

*Full code and data: [github.com/natwonglakhon/Taylor_Swift_Lyrics](https://github.com/natwonglakhon/Taylor_Swift_Lyrics)*
