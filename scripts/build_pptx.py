import os
import sys
import json
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import RESULTS_DIR, FIGURES_DIR

CSUB_BLUE = RGBColor(0x00, 0x35, 0x94)
CSUB_GOLD = RGBColor(0xFD, 0xB9, 0x13)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
INK       = RGBColor(0x20, 0x25, 0x35)
MUTED     = RGBColor(0x60, 0x68, 0x78)
PANEL     = RGBColor(0xF5, 0xF7, 0xFB)
GOLD_TINT = RGBColor(0xFF, 0xF4, 0xD0)

REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG   = FIGURES_DIR

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_bg(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.line.fill.background()
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    sp = bg._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)
    return bg


def add_bar(slide, x, y, w, h, color):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    return bar


def add_text(slide, x, y, w, h, text, *, size=18, bold=False, color=INK,
             align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0); tf.margin_right = Emu(0)
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def header(slide, title, subtitle=None):
    add_bar(slide, Inches(0), Inches(0.35), Inches(0.25), Inches(0.9), CSUB_GOLD)
    add_text(slide, Inches(0.55), Inches(0.3), Inches(12.2), Inches(0.8),
             title, size=30, bold=True, color=CSUB_BLUE)
    if subtitle:
        add_text(slide, Inches(0.55), Inches(1.05), Inches(12.2), Inches(0.5),
                 subtitle, size=16, color=MUTED)
    add_bar(slide, Inches(0), Inches(7.35), SLIDE_W, Inches(0.15), CSUB_BLUE)


# ----- slides --------------------------------------------------------------

def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, CSUB_BLUE)
    add_text(s, Inches(0.7), Inches(1.7), Inches(12.0), Inches(2.5),
             "House Price Prediction\nFrom Specs and Exterior Photos",
             size=50, bold=True, color=WHITE)
    add_bar(s, Inches(0.7), Inches(4.4), Inches(2.0), Inches(0.08), CSUB_GOLD)
    add_text(s, Inches(0.7), Inches(4.6), Inches(12.0), Inches(0.6),
             "CSU Bakersfield, CMPS 4450 Data Mining Final",
             size=22, color=CSUB_GOLD)
    add_text(s, Inches(0.7), Inches(5.3), Inches(12.0), Inches(0.6),
              "Noah Gallego, Gerardo Gomez, Justin Lo, Juancarlos Sandoval — Group 4",
              size=18, color=WHITE)


def slide_question(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "The Question",
           "Specs Already Get Us Most of the Way. Can the Photo Close the Gap?")
    add_text(s, Inches(0.55), Inches(2.0), Inches(12.2), Inches(0.5),
             "What We Know About Each House", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(2.55), Inches(12.2), Inches(1.5),
             "Beds, baths, square footage, city, and one exterior photo.",
             size=20, color=INK)
    add_text(s, Inches(0.55), Inches(4.3), Inches(12.2), Inches(0.5),
             "The Experiment", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(4.85), Inches(12.2), Inches(2.0),
             "Fit a model on specs alone. Then add features from the photo.\n"
             "Does the second model beat the first?",
             size=20, color=INK)


def slide_data(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Data",
           "15,474 Southern California Listings, $195k to $2M")
    s.shapes.add_picture(os.path.join(FIG, "fig01_price_hist.png"),
                         Inches(0.4), Inches(1.7), width=Inches(6.5))
    s.shapes.add_picture(os.path.join(FIG, "fig03_top_cities.png"),
                         Inches(7.0), Inches(1.7), width=Inches(6.0))


def slide_samples(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "What a Row Looks Like",
           "One Photo Plus a Few Numeric Fields Per House")
    s.shapes.add_picture(os.path.join(FIG, "fig02_sample_houses.png"),
                         Inches(1.7), Inches(1.55), height=Inches(5.5))


def slide_split(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Train, Validation, Test",
           "How We Use the 15,474 Listings")

    add_text(s, Inches(0.55), Inches(1.7), Inches(12.2), Inches(0.5),
             "How We Set It Up", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(2.15), Inches(12.2), Inches(3.5),
             "Shuffle the listings and split 65 / 15 / 20 into train, validation, and test\n"
             "Fit the models on train\n"
             "Pick the model and its settings (KNN k) by validation error\n"
             "Re-run the whole comparison on 500 random reshuffles to check stability\n"
             "Open the test set once, at the end, with every choice already locked",
             size=17, color=INK)

    add_text(s, Inches(0.55), Inches(5.2), Inches(12.2), Inches(0.5),
             "Why This Design", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(5.65), Inches(12.2), Inches(1.5),
             "One fixed split swings by several thousand dollars just from luck.\n"
             "The reshuffles smooth that out, and the test number stays honest because\n"
              "It never gets to inform a choice.",
             size=16, color=INK)


def slide_why_not_pixels(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Why Not Just Feed in the Pixels?",
           "The Dimensionality Wall")

    add_text(s, Inches(0.55), Inches(2.1), Inches(12.2), Inches(0.6),
              "Each Photo Is 128 x 128 x 3 = 49,152 Pixel Values",
              size=22, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(2.9), Inches(12.2), Inches(0.6),
              "Training Set Has Roughly 10,000 Houses",
              size=22, bold=True, color=CSUB_BLUE)

    add_text(s, Inches(0.55), Inches(4.2), Inches(12.2), Inches(0.6),
              "More Inputs Than Examples", size=20, bold=True, color=INK)
    add_text(s, Inches(0.55), Inches(4.8), Inches(12.2), Inches(2.5),
             "The model can fit anything in training and learn nothing useful.\n"
             "Linear regression is underdetermined.\n"
             "KNN sees pixel positions as unrelated columns — shift the house\n"
             "Ten pixels left and every input value changes.",
             size=17, color=INK)


def slide_filter_intro(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "A Filter Is a 3x3 Window That Slides Across the Photo",
           "The Kernel Is Fixed, the Output Is a Smaller Summary")
    gif_path = os.path.join(REPO, "slides_assets", "sobel_sliding.gif")
    if os.path.exists(gif_path):
        s.shapes.add_picture(gif_path, Inches(1.2), Inches(1.7), height=Inches(4.5))


def slide_filter_bank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Our Six Filters",
           "One Photo, Six Response Maps")
    s.shapes.add_picture(os.path.join(FIG, "fig19_filter_bank.png"),
                          Inches(0.2), Inches(1.9), width=Inches(13.0))



def slide_pooling(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Pooling: 128 x 128 Down to 4 x 4",
           "Each Filter Response Collapses to 16 Numbers")

    add_text(s, Inches(0.55), Inches(1.9), Inches(12.2), Inches(0.5),
             "The Idea", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(2.4), Inches(12.2), Inches(3.0),
             "Split each 128 x 128 response into a 4 x 4 grid of 32 x 32 patches.\n"
             "Take the mean of each patch.\n"
             "Now each filter is 16 numbers instead of 16,384.",
             size=18, color=INK)

    add_text(s, Inches(0.55), Inches(4.8), Inches(12.2), Inches(0.5),
             "What the Model Sees", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(5.3), Inches(12.2), Inches(2.0),
             "Six filters x 16 cells = 96 spatially-aware numbers per photo.\n"
             "Plus the 4 specs.  100 features total per house.",
             size=18, color=INK)


def slide_features(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "What the Model Sees, End to End",
           "100 Numbers Per House")

    add_text(s, Inches(0.55), Inches(2.0), Inches(6.0), Inches(0.6),
             "From the Listing", size=22, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(2.7), Inches(6.0), Inches(4.0),
             "Beds\nBaths\nSquare footage\nTypical price in that city",
             size=20, color=INK)

    add_text(s, Inches(7.0), Inches(2.0), Inches(6.0), Inches(0.6),
             "From the Photo", size=22, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(7.0), Inches(2.7), Inches(6.0), Inches(4.5),
             "6 filters, each pooled to 4 x 4\n"
             "= 96 numbers describing\n"
             "What's where in the photo",
             size=18, color=INK)


def _styled_table(slide, x, y, w, h, rows, *, header_fill=None, body_fill=None,
                  highlight_rows=(), highlight_fill=None, font_size=13, header_font_size=14):
    if header_fill is None: header_fill = CSUB_BLUE
    if body_fill   is None: body_fill   = WHITE
    if highlight_fill is None: highlight_fill = GOLD_TINT
    n_rows = len(rows); n_cols = len(rows[0])
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    tbl = tbl_shape.table
    for ci in range(n_cols):
        for ri in range(n_rows):
            cell = tbl.cell(ri, ci)
            cell.text = ""
            tf = cell.text_frame
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(rows[ri][ci])
            r.font.name = "Calibri"
            r.font.size = Pt(header_font_size if ri == 0 else font_size)
            r.font.bold = (ri == 0)
            if ri == 0:
                r.font.color.rgb = WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            else:
                r.font.color.rgb = INK
                cell.fill.solid()
                if ri in highlight_rows:
                    cell.fill.fore_color.rgb = highlight_fill
                else:
                    cell.fill.fore_color.rgb = body_fill if ri % 2 == 1 else PANEL
    return tbl


def slide_columns(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Eight Columns From Kaggle",
           "What Arrived in the CSV, and What We Do With Each One")

    rows = [
        ["Column",      "Example",                "Action"],
        ["image_id",    "0",                      "Drop"],
        ["street",      "1317 Van Buren Avenue",  "Drop"],
        ["citi",        "Salton City, CA",        "Convert to coordinates"],
        ["n_citi",      "317",                    "Drop"],
        ["bed",         "3",                      "Keep"],
        ["bath",        "2.0",                    "Keep"],
        ["sqft",        "1,560",                  "Keep"],
        ["price",       "$201,900",               "Target"],
    ]
    _styled_table(s, Inches(0.55), Inches(1.7), Inches(12.2), Inches(4.8),
                  rows, font_size=14, header_font_size=15)


def slide_encoding(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "City Names Become Coordinates",
           "We Look Each City Up Once With a Geocoding API")

    add_text(s, Inches(0.55), Inches(1.9), Inches(12.2), Inches(0.6),
             "The Lookup", size=20, bold=True, color=CSUB_BLUE)
    lookup_rows = [
        ["City",              "Latitude",  "Longitude"],
        ["Beverly Hills, CA", "34.0736",   "-118.4004"],
        ["San Diego, CA",     "32.7157",   "-117.1611"],
        ["Bakersfield, CA",   "35.3733",   "-119.0187"],
        ["Salton City, CA",   "33.3030",   "-115.9486"],
    ]
    _styled_table(s, Inches(0.55), Inches(2.5), Inches(12.2), Inches(2.3),
                  lookup_rows, font_size=14, header_font_size=15)

    add_text(s, Inches(0.55), Inches(5.0), Inches(12.2), Inches(0.6),
             "Why This Works", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(0.55), Inches(5.55), Inches(12.2), Inches(2.0),
             "415 unique cities  =>  415 API calls, cached once on disk.\n"
             "Every row gets its city's latitude and longitude.\n"
             "Now 'nearest neighbor' actually means geographically near, not just same city name.",
             size=16, color=INK)


def slide_why_k10(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Why k=10?",
            "Elbow Method on 415 City Coordinates")
    img = os.path.join(FIG, "fig22_elbow.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.4), Inches(1.7), width=Inches(8.5))


def slide_regions(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Grouping Cities Into Regions",
           "K-Means on the 415 City Coordinates, k=10")
    img = os.path.join(FIG, "fig22_regions.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.4), Inches(1.6), height=Inches(5.0))
    add_text(s, Inches(9.6), Inches(2.0), Inches(3.4), Inches(0.5),
             "The Idea", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(9.6), Inches(2.6), Inches(3.4), Inches(4.5),
             "K-means takes the\n"
             "Coordinates and groups\n"
             "Nearby cities together.\n\n"
             "Every row gets one\n"
             "Extra feature: region_id.",
             size=14, color=INK)


def slide_normalization(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Normalize So No Column Dominates",
           "Rescale Every Feature to the Same Range")

    rows = [
        ["Feature",    "Raw Range",            "After Min-Max"],
        ["bed",        "1 to 6",               "0 to 1"],
        ["bath",       "1 to 5",               "0 to 1"],
        ["sqft",       "500 to 5,400",         "0 to 1"],
        ["city_lat",   "32.5 to 35.4",         "0 to 1"],
        ["city_lon",   "-119.5 to -114.5",     "0 to 1"],
        ["region_id",  "0 to 9",               "0 to 1"],
    ]
    _styled_table(s, Inches(0.55), Inches(1.9), Inches(12.2), Inches(3.9),
                  rows, font_size=14, header_font_size=15)


def slide_three_models(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Two Models",
           "To Establish a Text-Only Baseline Before We Touch the Photos")

    ys = Inches(1.9); col_w = Inches(6.0); gap = Inches(0.4)
    xs = [Inches(0.55), Inches(0.55) + col_w + gap]
    titles = ["Multiple Linear Regression", "KNN Regressor"]
    bodies = [
        "One weight per feature, all summed plus an intercept.\n\n"
        "Prediction =\nw1*bed + w2*bath + w3*sqft\n"
        "+ w4*city_lat + w5*city_lon + w6*region.\n\n"
        "Fast and transparent. Assumes the relationship\nis a straight line.",

        "For a new house, find the k closest houses in\nfeature space.\n\n"
        "Predict by averaging their prices\n(regression, not voting).\n\n"
        "Needs normalized features. Tuned by k\nand by the distance function.",
    ]
    for x, t, b in zip(xs, titles, bodies):
        add_bar(s, x, ys, col_w, Inches(0.6), CSUB_BLUE)
        add_text(s, x + Inches(0.2), ys + Inches(0.1), col_w, Inches(0.5),
                 t, size=20, bold=True, color=WHITE)
        add_text(s, x + Inches(0.2), ys + Inches(0.95), col_w - Inches(0.4), Inches(5.0),
                 b, size=15, color=INK)


def slide_mlr_results(prs, mlr):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Multiple Linear Regression",
           "Validation Performance, Predicted vs Actual Price")
    img = os.path.join(FIG, "fig23_mlr_pred_vs_actual.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.3), Inches(1.55), height=Inches(5.6))

    m = mlr["validation_metrics"]
    box_x = Inches(8.5); box_y = Inches(1.9)
    add_text(s, box_x, box_y, Inches(4.4), Inches(0.5),
             "Results", size=22, bold=True, color=CSUB_BLUE)

    rows = [
        ("MAE",   f"${m['mae']/1000:,.0f}k"),
        ("RMSE",  f"${m['rmse']/1000:,.0f}k"),
    ]
    y = box_y + Inches(0.85)
    for label, val in rows:
        add_text(s, box_x, y, Inches(2.4), Inches(0.6),
                 label, size=22, color=MUTED)
        add_text(s, box_x + Inches(2.4), y, Inches(2.0), Inches(0.6),
                 val, size=26, bold=True, color=INK)
        y = y + Inches(0.9)


def slide_knn_euclidean(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "KNN With Euclidean Distance",
           "Each Box Summarizes 500 Random Reshuffles at That k")
    img = os.path.join(FIG, "fig24_knn_euclidean.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.5), Inches(1.7), width=Inches(12.3))


def slide_knn_distances(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "We Repeated This for All Four Distance Functions",
           "Euclidean, Manhattan, Chebyshev, Minkowski")
    img = os.path.join(FIG, "fig20_knn_distances.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.3), Inches(1.6), width=Inches(12.7))


def slide_baseline_box(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Specs Only Is Already Strong",
           "Error in Dollars, Lower Is Better")
    img = os.path.join(FIG, "fig05_baseline_boxplot.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.4), Inches(1.7), width=Inches(8.8))
    add_text(s, Inches(9.5), Inches(2.0), Inches(3.5), Inches(0.6),
             "Specs-Only Baseline", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(9.5), Inches(2.7), Inches(3.5), Inches(4.0),
             "500 reshuffles,\nspecs-only models.\n\n"
             "KNN around $152k MAE.\nLinear around $164k.\n\n"
             "Both are well above\nthe constant predictor\nat $280k.",
             size=16, color=INK)


def slide_worst_misses(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Where the Baseline Misses",
           "The Top Six Errors Split Into Two Kinds")
    s.shapes.add_picture(os.path.join(FIG, "fig07_worst_misses.png"),
                         Inches(1.8), Inches(1.55), height=Inches(5.6))


def slide_cap_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "The Cap Problem",
           "The Dataset Has a Ceiling")
    s.shapes.add_picture(os.path.join(FIG, "fig18_cap_cluster.png"),
                          Inches(1.4), Inches(1.7), width=Inches(10.5))



def slide_hypothesis(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Three Things the Photo Could Fix",
           "The Case for Adding Image Features")

    ys = Inches(2.2); col_w = Inches(4.0); gap = Inches(0.3)
    xs = [Inches(0.55), Inches(0.55) + col_w + gap, Inches(0.55) + 2 * (col_w + gap)]
    titles = ["Condition", "Size Beyond Sqft", "Neighborhood Look"]
    bodies = [
        "Same specs, different\nupkeep and landscaping.\nThat lives in the photo.",
        "Story count, lot size,\nyard space. Specs miss\nthose. The photo doesn't.",
        "Trees, driveway, fencing,\nlighting. Signals the\nstreet, not just the city.",
    ]
    for x, t, b in zip(xs, titles, bodies):
        add_bar(s, x, ys, col_w, Inches(0.55), CSUB_BLUE)
        add_text(s, x + Inches(0.15), ys + Inches(0.07), col_w, Inches(0.5),
                 t, size=20, bold=True, color=WHITE)
        add_text(s, x + Inches(0.15), ys + Inches(0.85), col_w - Inches(0.3), Inches(3.0),
                 b, size=17, color=INK)


def slide_image_box(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Adding Photo Features",
           "Same Two Models, With and Without the Photo")
    img = os.path.join(FIG, "fig11_specs_vs_image_boxplot.png")
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.4), Inches(1.7), height=Inches(4.6))


def slide_bad_photos(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Why? Some Photos Aren't of the House",
           "Examples Pulled From the Worst-Miss List")
    s.shapes.add_picture(os.path.join(FIG, "fig16_bad_photos.png"),
                          Inches(0.3), Inches(2.2), width=Inches(12.7))



def slide_feature_scales(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Why? Feature Scales Are Wildly Different",
           "Raw Values Span Orders of Magnitude")
    s.shapes.add_picture(os.path.join(FIG, "fig17_feature_scales.png"),
                          Inches(0.3), Inches(1.55), height=Inches(5.0))



def slide_preprocess(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "What We Tried",
           "Three Preprocessing Fixes, Each Motivated by a Problem Above")

    rows = [
        ("Center Crop",          "Throw out the outer 20% on every side, focus on what's in the middle of the frame.",
                                 "Addresses photos where sky or pavement dominates."),
        ("Per-Image Normalize",  "Rescale each photo so its mean brightness matches every other photo.",
                                 "Addresses photos shot at different times of day or different exposures."),
        ("Add HSV Channels",     "Compute hue, saturation, and value averages on top of RGB.",
                                 "Color signal that doesn't change with brightness."),
    ]
    y = Inches(2.0)
    for name, what, why in rows:
        add_text(s, Inches(0.55), y, Inches(3.0), Inches(0.5),
                 name, size=20, bold=True, color=CSUB_BLUE)
        add_text(s, Inches(3.8), y, Inches(9.0), Inches(0.5),
                 what, size=15, color=INK)
        add_text(s, Inches(3.8), y + Inches(0.45), Inches(9.0), Inches(0.5),
                 why, size=14, color=MUTED)
        y = y + Inches(1.3)


def slide_variant_table(prs, variants):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Did Preprocessing Help?",
           "Validation MAE by Feature Variant, Linear and KNN")

    left = Inches(0.7); top = Inches(2.0)
    widths = [Inches(4.6), Inches(2.4), Inches(2.4), Inches(2.4)]
    cell_h = Inches(0.55)
    heads = ["Feature Variant", "Linear MAE", "KNN MAE", "Best"]

    def cell(x, y, w, h, text, *, fill, color, bold=False, size=14):
        shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        shp.line.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
        tf = shp.text_frame
        tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text
        r.font.name = "Calibri"; r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color

    x = left
    for w, h in zip(widths, heads):
        cell(x, top, w, cell_h, h, fill=CSUB_BLUE, color=WHITE, bold=True, size=15)
        x += w

    rows_data = [
        ("Specs Only",                "specs_only_linear",          "specs_only_knn"),
        ("Specs + Raw Photo",         "specs_+_raw_image_linear",   "specs_+_raw_image_knn"),
        ("Specs + Center Crop",       "specs_+_crop_linear",        "specs_+_crop_knn"),
        ("Specs + Brightness Norm",   "specs_+_per-img_norm_linear","specs_+_per-img_norm_knn"),
        ("Specs + Crop + Norm + HSV", "specs_+_crop+norm+HSV_linear","specs_+_crop+norm+HSV_knn"),
    ]
    all_bests = []
    for _, lk, kk in rows_data:
        lm = variants[lk]["mae"]; km = variants[kk]["mae"]
        all_bests.append(min(lm, km))
    overall_best = min(all_bests)
    y = top + cell_h
    for i, (label, lk, kk) in enumerate(rows_data):
        lm = variants[lk]["mae"]; km = variants[kk]["mae"]
        best = min(lm, km)
        bg = WHITE if i % 2 == 0 else PANEL
        if best == overall_best:
            bg = GOLD_TINT
        x = left
        cell(x, y, widths[0], cell_h, label,                   fill=bg, color=INK, size=14); x += widths[0]
        cell(x, y, widths[1], cell_h, f"${lm/1000:.0f}k",      fill=bg, color=INK, size=14); x += widths[1]
        cell(x, y, widths[2], cell_h, f"${km/1000:.0f}k",      fill=bg, color=INK, size=14); x += widths[2]
        cell(x, y, widths[3], cell_h, f"${best/1000:.0f}k",    fill=bg, color=INK, bold=True, size=14)
        y += cell_h


def slide_spatial_results(prs, spatial):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Did Spatial Features Help?",
           "Validation MAE, by Feature Set and Model")

    left = Inches(0.8); top = Inches(2.0)
    widths = [Inches(4.8), Inches(2.2), Inches(2.2), Inches(2.0)]
    cell_h = Inches(0.55)
    heads = ["Feature Set", "Linear", "KNN", "Best"]

    def cell(x, y, w, h, text, *, fill, color, bold=False, size=14):
        shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        shp.line.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
        tf = shp.text_frame
        tf.margin_left = Inches(0.12); tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text
        r.font.name = "Calibri"; r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color

    x = left
    for w, h in zip(widths, heads):
        cell(x, top, w, cell_h, h, fill=CSUB_BLUE, color=WHITE, bold=True, size=14)
        x += w

    rows = [
        ("Specs Only",            "specs_only"),
        ("Specs + Spatial (96)",  "specs_+_spatial_96"),
        ("Spatial Only",          "spatial_only_no_specs"),
    ]
    all_bests = []
    for _, prefix in rows:
        lin  = spatial[f"{prefix}_linear"]["mae"]
        knn  = spatial[f"{prefix}_knn"]["mae"]
        all_bests.append(min(lin, knn))
    overall_best = min(all_bests)
    y = top + cell_h
    for i, (label, prefix) in enumerate(rows):
        lin  = spatial[f"{prefix}_linear"]["mae"]
        knn  = spatial[f"{prefix}_knn"]["mae"]
        best = min(lin, knn)
        bg = WHITE if i % 2 == 0 else PANEL
        if best == overall_best and prefix == "specs_only":
            bg = GOLD_TINT
        x = left
        cell(x, y, widths[0], cell_h, label, fill=bg, color=INK, size=14); x += widths[0]
        cell(x, y, widths[1], cell_h, f"${lin/1000:.0f}k",  fill=bg, color=INK, size=14); x += widths[1]
        cell(x, y, widths[2], cell_h, f"${knn/1000:.0f}k",  fill=bg, color=INK, size=14); x += widths[2]
        cell(x, y, widths[3], cell_h, f"${best/1000:.0f}k", fill=bg, color=INK, bold=True, size=14)
        y += cell_h


def slide_subgroup(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Per-City, With and Without the Photo",
           "KNN, Specs Only vs Specs + Image. Two Cities Benefit a Little. The Rest Get Worse.")
    s.shapes.add_picture(os.path.join(FIG, "fig12_per_city_delta.png"),
                         Inches(0.6), Inches(1.6), height=Inches(5.5))


def slide_kmeans(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "K-Means on Simple Image Features",
            "Clustering on the 27 Color and Edge Stats, Not the Spatial Filters")
    s.shapes.add_picture(os.path.join(FIG, "fig14_kmeans_image.png"),
                         Inches(0.3), Inches(1.6), width=Inches(9.0))
    add_text(s, Inches(9.6), Inches(2.0), Inches(3.4), Inches(0.6),
             "What We Found", size=20, bold=True, color=CSUB_BLUE)
    add_text(s, Inches(9.6), Inches(2.7), Inches(3.4), Inches(4.0),
             "Four clusters, mean\nprices land within\n$80k of each other.\n\n"
             "The photo features\ndon't separate price.",
             size=16, color=INK)


def slide_test_table(prs, m):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, WHITE)
    header(s, "Final Test Results",
           "Trained on Train + Val, Evaluated Once on the 3,095 Held-Out Test Rows")

    left = Inches(0.7); top = Inches(1.9)
    widths = [Inches(3.4), Inches(2.4), Inches(1.7), Inches(1.7), Inches(1.5), Inches(1.5)]
    cell_h = Inches(0.5)
    heads = ["Model", "Features", "MAE", "RMSE", "% Within $50k", "% Within $100k"]

    def cell(x, y, w, h, text, *, fill, color, bold=False, size=13):
        shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
        shp.line.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
        tf = shp.text_frame
        tf.margin_left = Inches(0.1); tf.margin_right = Inches(0.1)
        tf.margin_top = Inches(0.03); tf.margin_bottom = Inches(0.03)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = text
        r.font.name = "Calibri"; r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color

    x = left
    for w, h in zip(widths, heads):
        cell(x, top, w, cell_h, h, fill=CSUB_BLUE, color=WHITE, bold=True, size=14)
        x += w

    rows = [
        ("Constant",                  "None",            m["test_constant_predictor"]),
        ("Linear Regression",         "Specs",           m["test_linear_specs"]),
        ("Linear Regression",         "Specs + Spatial", m["test_linear_specs_spatial"]),
        ("KNN, k=15",                 "Specs",           m["test_knn_specs_k15"]),
        ("KNN, k=15",                 "Specs + Spatial", m["test_knn_specs_spatial_k15"]),
    ]
    best_idx = min(range(len(rows)), key=lambda i: rows[i][2]["mae"])
    y = top + cell_h
    for i, (name, feat, met) in enumerate(rows):
        bg = WHITE if i % 2 == 0 else PANEL
        if i == best_idx:
            bg = GOLD_TINT
        x = left
        cell(x, y, widths[0], cell_h, name, fill=bg, color=INK, size=12); x += widths[0]
        cell(x, y, widths[1], cell_h, feat, fill=bg, color=INK, size=12); x += widths[1]
        cell(x, y, widths[2], cell_h, f"${met['mae']/1000:.0f}k",  fill=bg, color=INK, size=12); x += widths[2]
        cell(x, y, widths[3], cell_h, f"${met['rmse']/1000:.0f}k", fill=bg, color=INK, size=12); x += widths[3]
        cell(x, y, widths[4], cell_h, f"{met['within_50k']:.1%}",  fill=bg, color=INK, size=12); x += widths[4]
        cell(x, y, widths[5], cell_h, f"{met['within_100k']:.1%}", fill=bg, color=INK, size=12)
        y += cell_h


def slide_conclusion(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, CSUB_BLUE)
    add_bar(s, Inches(0), Inches(1.2), SLIDE_W, Inches(0.08), CSUB_GOLD)
    add_text(s, Inches(0.6), Inches(0.5), Inches(12.2), Inches(0.7),
             "Conclusion", size=34, bold=True, color=WHITE)

    add_text(s, Inches(0.6), Inches(1.6), Inches(12.2), Inches(1.4),
              "Specs already carry the price signal.\n"
              "The photo, hand-summarized in numpy, does not add anything on top.",
             size=24, color=CSUB_GOLD)

    add_text(s, Inches(0.6), Inches(4.0), Inches(12.2), Inches(0.6),
              "What We Tried", size=20, bold=True, color=CSUB_GOLD)
    add_text(s, Inches(0.6), Inches(4.55), Inches(12.2), Inches(3.0),
              "27 simple image features and 96 spatial filter features.\n"
              "Preprocessing (crop, normalize, HSV) didn't help.\n"
              "Neither linear nor KNN improved with photo features added.",
             size=18, color=WHITE)


def main():
    with open(os.path.join(RESULTS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "spatial_comparison.json")) as f:
        spatial = json.load(f)
    with open(os.path.join(RESULTS_DIR, "mlr_v3.json")) as f:
        mlr = json.load(f)
    with open(os.path.join(RESULTS_DIR, "variant_comparison.json")) as f:
        variants = json.load(f)

    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_title(prs)
    slide_question(prs)
    slide_data(prs)
    slide_samples(prs)
    slide_split(prs)
    slide_columns(prs)
    slide_encoding(prs)
    slide_why_k10(prs)
    slide_regions(prs)
    slide_normalization(prs)
    slide_three_models(prs)
    slide_mlr_results(prs, mlr)
    slide_knn_euclidean(prs)
    slide_knn_distances(prs)
    slide_baseline_box(prs)
    slide_worst_misses(prs)
    slide_hypothesis(prs)
    slide_image_box(prs)
    slide_preprocess(prs)
    slide_variant_table(prs, variants)
    slide_why_not_pixels(prs)
    slide_filter_intro(prs)
    slide_filter_bank(prs)
    slide_features(prs)
    slide_spatial_results(prs, spatial)
    slide_subgroup(prs)
    slide_test_table(prs, metrics)
    slide_conclusion(prs)

    out = os.path.join(REPO, "HousePrice_Final_Project.pptx")
    try:
        prs.save(out)
    except PermissionError:
        v = 2
        while True:
            out = os.path.join(REPO, f"HousePrice_Final_Project_v{v}.pptx")
            try:
                prs.save(out)
                break
            except PermissionError:
                v += 1
                if v > 20:
                    out = os.path.join(os.environ.get("TEMP", "/tmp"), "HousePrice_Final_Project.pptx")
                    prs.save(out)
                    break
    print(f"saved {out}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
