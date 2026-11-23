# CWA-Based MesoViewer

A Streamlit web app that plots SPC SFCOA-HRRR mesoanalysis fields for any NWS CWA or the full CONUS.
Data is pulled live from NOMADS (`spc_post/prod/.../sfcoa-hrrr/`) and refreshed every 5 minutes.

## Files

| File | What it is |
|---|---|
| `app.py` | The whole app |
| `requirements.txt` | Python packages the host installs |
| `.gitignore` | Keeps downloaded `.nc` data files out of the repo |

## Deploy on Streamlit Community Cloud (free)

1. Create a new GitHub repository and upload these three files to it (README optional).
2. Sign in to Streamlit Community Cloud with your GitHub account.
3. Create a new app and point it at the repository, branch `main`, main file `app.py`.
4. In the advanced settings, choose Python 3.12 (or the newest version offered).
5. Deploy. The first build takes several minutes while cartopy, MetPy and the rest install.

The app gets an address like `your-app-name.streamlit.app`.

## Embedding in a Google Site

In Google Sites, use **Insert → Embed → By URL** with the app address plus `?embed=true`,
for example `https://your-app-name.streamlit.app/?embed=true`. The app itself stays public.

## Watching memory

Under the map, the app shows its current memory, its peak memory since startup, and the size of
the data files on disk. If the peak gets close to the host's limit, the easiest settings to lower
are near the top of `app.py`:

- `MAX_PAST_HOURS` (currently 3): how many past analyses the slider offers
- `RENDER_DPI` (currently 150): image sharpness; lower means smaller images
- `STRIP_MAX_POINTS` (currently 60,000): size of the row strips used for 3D calculations
