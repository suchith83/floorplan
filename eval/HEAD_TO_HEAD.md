# Head-to-head vs a consumer scanning app

**Status: NOT DONE.** Both rows below are open, for different reasons.

| comparison | status | reason |
|---|---|---|
| Brief's version: our **LiDAR tier** vs a consumer app on 2 of our own benchmark rooms | **Not done (no iPhone Pro; recruiter-approved)** | The user has no LiDAR iPhone. The recruiter said to skip own LiDAR captures and run the LiDAR tier on her sample data, which has no app export and no tape. |
| Substitute: our **photo / video tier** vs magicplan (free tier) on the same 2 rooms, both against tape | **Not done: own captures not taken** | Needs `data/own/` (captures, tape, app export). None of it exists yet (8 Oct). |

The substitute is not what the brief asks for: it compares our weakest tiers with an app on a phone without LiDAR.
It is labelled as a substitute everywhere it appears, and the compliance matrix keeps the LiDAR row as Not done.

## Protocol for the substitute (ready to run)

1. On the user's phone, scan the **same 2 rooms** that were taped (CHECKLIST Phase 2e) with **magicplan, free tier**.
   Note the app version (Settings → About). Export or screenshot every dimension into `data/own/app_export/`
   (`magicplan_<room>.pdf` / `.png`, plus `app.txt` with the app name, version, phone model and date).
2. Type the app's numbers into `eval/ground_truth/app_magicplan.yaml` using the **same format and the same wall
   names** as the tape file (`eval/ground_truth/TEMPLATE.yaml`; readings are single values).
3. Our side: `make benchmark` scores `own-photos_run1` and `own-video_run1` against tape (`eval/benchmark.json`,
   `own_scored`).
4. Table, dimension by dimension, for every dimension both the app and we report (wall lengths, door widths,
   ceiling if the app gives one):

| room | dimension | tape | magicplan | |err| app | ours (tier) | |err| ours | beat or tie? |
|---|---|---|---|---|---|---|---|
| _not done: own captures not taken_ | | | | | | | |

5. **Beat or tie** = our |error| ≤ the app's |error| + 0.5 cm (a tape reading's own resolution). Target: ≥ 70 % of
   shared dimensions. Reported as measured, including if we lose.

## What to expect (honest prior)

On the sample-derived inputs our camera tiers are far from the LiDAR plan (median wall error 0.4–1.3 m,
`eval/BENCHMARK.md`). magicplan's photo-and-AR flow on a non-LiDAR phone will likely win most dimensions.
That comparison is still worth running, because it is the one that shows how far the photo tier is from a product.
