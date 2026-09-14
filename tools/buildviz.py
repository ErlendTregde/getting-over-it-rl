"""Fold the captured run and the pot cutout into the standalone pages.

Both pages ship as one file each: the artifact host allows no external
fetches, so the run data and the PNG are embedded rather than linked.

    uv run .\buildviz.py
"""
import base64
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def build(tpl_name, out_name, data_file=None, png_file=None):
    p = os.path.join(HERE, tpl_name)
    s = io.open(p, encoding="utf-8").read()
    if data_file:
        d = io.open(os.path.join(HERE, data_file), encoding="utf-8").read()
        assert s.count("/*__DATA__*/") == 1, tpl_name
        s = s.replace("/*__DATA__*/", d, 1)
    if png_file:
        b = base64.b64encode(open(os.path.join(HERE, png_file), "rb").read())
        assert s.count("/*__PLAYER__*/") == 1, tpl_name
        s = s.replace("/*__PLAYER__*/",
                      "data:image/png;base64," + b.decode("ascii"), 1)
    out = os.path.join(HERE, out_name)
    io.open(out, "w", encoding="utf-8").write(s)
    print(f"  {out_name:16} {os.path.getsize(out)/1024:6.0f} KB")


if __name__ == "__main__":
    build("netviz_template.html", "netviz.html", "netcap.json", "player.png")
    build("netlive_template.html", "netlive.html", "netcap.json")
    build("netsync_template.html", "netsync.html", "netcap.json")
    build("netfull_template.html", "netfull.html", "netcap.json", "player.png")
    build("netreveal.html", "netreveal.html")   # standalone, nothing to inject
