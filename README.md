# AI Engineers Handbook

Deep-dive notes for AI engineers, built with [Zensical](https://zensical.org/) and deployed to GitHub Pages at <https://nithish96.github.io/>.

## Run locally

Requires Python 3.10 or newer and Node.js 22 or newer. KaTeX is installed from the pinned npm dependency and copied into the documentation assets before building.

```sh
npm ci
npm run assets:katex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
zensical serve
```

Open <http://localhost:8000/> to preview the site. Build the static site with `zensical build`; generated files are written to `site/`.

## Deploy

The GitHub Actions workflow builds and deploys the site whenever changes are pushed to `main` (or when manually triggered). In the repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**.
