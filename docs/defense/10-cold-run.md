# 10 — Cold run: defense notes

- **How long from a fresh clone to a plan?** 2 min 10 s on my Mac with warm caches: clone, `uv sync`, zips into
  `data/zips/`, `scripts/fetch_data.sh`, `fp run` on the single room (112 s of it is the run). A clean machine adds the
  ~2 GB wheel download and 0.9 GB of damage weights; I did not time that.
- **Where does the data come from?** Your sample zips. I did not publish a dataset; `fetch_data.sh` tries one, fails,
  warns, and unpacks `data/zips/`.
- **What did the rehearsal find?** One crash. A scan that never revisits a place has no loop to score, the loop RMSE was
  NaN, and NaN is not valid JSON, so the plan failed validation. The sample scans all revisit, so the benchmark never
  hit it. Now it is null, meaning not measured. Found with a scan cut to its first 40 %.
- **What happens with too few photos?** Two per room with no doorway photo: the rooms that share nothing are not placed,
  and the warning says to re-capture with a doorway photo in both folders. It does not guess.
- **Junk in the folder?** Non-image files are ignored; a broken `.JPG` or empty `.heic` is skipped with a warning.
- **What didn't you test?** Mixed HEIC/JPEG, a video with fast pans, darkness and a mirror, and a truly cold machine.
