import logging

import matplotlib.pyplot as plt

from gibbsq.analysis import plotting


def test_apply_theme_uses_installed_sans_serif_font(caplog):
    caplog.set_level(logging.WARNING, logger="matplotlib.font_manager")

    plotting._apply_theme()

    fig, ax = plt.subplots()
    ax.set_title("Font smoke test")
    fig.canvas.draw()
    plt.close(fig)

    assert "Generic family 'sans-serif' not found" not in caplog.text
    assert "DejaVu Sans" in plt.rcParams["font.sans-serif"]
