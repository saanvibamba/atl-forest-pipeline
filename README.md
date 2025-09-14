1. Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install numpy opencv-python scikit-image scikit-learn matplotlib ipywidgets ipympl jupyterlab
# pip install notebook

2. Add images into data/raw

3. What the pipeline does:
    Environment & Paths (Cell 0–1)
        Prints versions (NumPy/OpenCV/skimage/sklearn), initializes data/ and results/.
        Two toggles via env vars:
        USE_WIDGETS=1 → interactive matplotlib widgets
        ENABLE_ACTIVE=1 → optional active-learning flow (Cells 16–18)
    Load & Preprocess (Cells 2–3)
        Load RGB, convert to grayscale, resize to 512×512, blur, histogram equalize.
        Saves preprocessed images to data/processed.
    Baselines (Cells 4–7,8)
        Otsu thresholding (with auto flip if vegetation is black).
        Canny + morphology to capture textured/edge-rich regions.
        K-Means (k=2) on an 8-feature stack:
        gray, normalized R/G/B, vegetation proxy (G−R)/(G+R), Laplacian, gradient magnitude, local entropy/variance
        Visual side-by-side panels and forest fraction per method.
    Weak labels (Cells 9–10)
        Auto-generate balanced clicks per image (60 pos / 60 neg) from baseline agreement vs. disagreement interiors (away from borders).
        Merge/dedupe and build training arrays with per-source weights.
    Random Forest & Per-image Thresholding (Cell 11)
        Train RandomForestClassifier with GroupShuffleSplit by image (prevents leakage).
        Pick a global operating point from held-out F1, then for each image binary-search a per-image threshold on the RF probabilities so that the post-cleaned mask matches a target coverage estimated from baselines (clamped to [0.45, 0.60]).
        Save RF masks (post-clean) to results/random_forest_hybrid/.
    Refine & Visual QA (Cell 12–14)
        Refinement: trim borders, light close/open, fill small holes, area-filtering.
        Save refined masks to results/random_forest_refined/ and quick-peek panels to /figures/.
        Optional paper/border suppression for scanned imagery and color overlays for inspection.
    Final Comparison (Cell 15)
        Print fractions and IoU vs RF* (RF refined) for baselines and RF(raw).
        Show 5-panel figures: Original, Otsu, Canny, KMeans, RF refined.
    (Optional) Active Learning (Cells 16–18, gated by ENABLE_ACTIVE)
        Propose uncertain points (|p−0.5| small) with spacing constraints.
        Simple widget to label points, then retrain with higher weights for new labels and reapply.

4. How to run:
    Start Jupyter, open the notebook.
    Cell-by-cell:
        Run Cell 0–3 (setup + preprocess).
        Run Cells 4–8 (baselines + fractions).
        Run Cells 9–12 (auto clicks → train/apply → refine → figures).
        Run Cell 13–15 (overlays, border suppression, final comparison).
        (Optional) Set ENABLE_ACTIVE=1 and run Cells 16–18.
    After a first full run, you can re-run only Cells 9–12 to retrain and update masks.# atl-forest-pipeline
