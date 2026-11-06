# Profile Banner

`profile.png` (repo root) is the banner at the top of my GitHub profile. It's drawn by
[`make_banner.py`](make_banner.py) at 3x size (2874x627) from the design in
[`pattern.json`](pattern.json), so it stays sharp on high-DPI screens.

## Changing the Text

Get the font (OFL-licensed, from the official Google Fonts repo). It isn't committed here:

```bash
curl -sSfL -o /tmp/RedHatDisplay.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/redhatdisplay/RedHatDisplay%5Bwght%5D.ttf"
```

Then, from the repo root (`uv` reads the dependencies from the script header):

```bash
uv run banner/make_banner.py profile.png --font /tmp/RedHatDisplay.ttf --subtitle "Data Engineer"
```

| Option | Default | What it does |
| --- | --- | --- |
| `--title` | `Dominic Bean` | First line |
| `--subtitle` | `Data Engineer` | Second line |
| `--scale` | `3` | Size relative to the 958x209 original |
| `--supersample` | `4` | Anti-aliasing: everything is drawn this much bigger, then averaged down |

Both lines are centred, so text of any length works.

## Where the Design Comes From

The 2022 banner came from an online generator as a 958x209 PNG with no source file, which is
why it looked soft on high-DPI screens. [`extract_pattern.py`](extract_pattern.py) recovered it
as geometry, once:

- **Background:** flat `#2C82C0`, 8px rounded corners.
- **Network pattern:** 37 visible dots, 3 more hidden behind the name, and 154 lines (including
  those that run off the edges). Lines are 1.6px wide at about 5% white; dots have a 1.55px
  radius at about 28% white. Widths and opacities were calibrated by rendering and comparing.
- **Text:** Red Hat Display. The name is weight 400 at 41.8px; the subtitle weight 392 at
  20.9px.

Check: rendered at 1x with the original "Data Scientist" subtitle, the redraw matches the 2022
banner to within 0.11 (out of 255) on average across the pattern.

To re-run the extraction (only needed if the design is lost or changed):

```bash
git show 15348b6:profile.png > /tmp/original.png
uv run banner/extract_pattern.py /tmp/original.png banner/pattern.json --font /tmp/RedHatDisplay.ttf
```

## History

| Date | Change |
| --- | --- |
| 2022-09-27 | Banner added ("Dominic Bean" / "Data Scientist"), 958x209 |
| 2026-11-06 | Subtitle changed to "Data Engineer" by editing the pixels |
| 2026-11-06 | Redrawn from recovered geometry at 3x (2874x627) to fix the blur |
