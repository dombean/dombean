# Profile Banner

`profile.png` (repo root) is the banner at the top of my GitHub profile: white text on a blue
patterned background, set in **Red Hat Display**. The text is part of the image, so changing it
means editing pixels. [`retitle_banner.py`](retitle_banner.py) does that cleanly.

## What the Script Does

1. Finds the old text inside a band of rows (by default the subtitle, rows 120-170).
2. Fits the font's weight, size and position to the existing pixels.
3. Inpaints the old text away, so the background lines carry on through.
4. Draws the new text with the fitted settings, centred where the old text was centred and on the
   same baseline.

Only the text area changes. The rest of the image, including the transparent rounded corners, is
left exactly as it was.

## Usage

Get the font (OFL-licensed, from the official Google Fonts repo). It is not committed here:

```bash
curl -sSfL -o /tmp/RedHatDisplay.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/redhatdisplay/RedHatDisplay%5Bwght%5D.ttf"
```

Run from the repo root. `uv` reads the dependencies from the script header:

```bash
uv run banner/retitle_banner.py profile.png profile.png --font /tmp/RedHatDisplay.ttf --old "Data Engineer" --new "Senior Data Engineer"
```

- `--old` must match the text on the banner exactly.
- `--band TOP BOTTOM` picks a different line: the name sits at roughly rows 60-110.
- Check the result before committing: zoom in on the text and compare with the previous version
  (`git show HEAD:profile.png > /tmp/old.png`).

## History

| Date | Change |
| --- | --- |
| 2022-09-27 | Banner added ("Dominic Bean" / "Data Scientist") |
| 2026-11-06 | Subtitle changed to "Data Engineer" with this script. Fitted: weight 350, 21px |
