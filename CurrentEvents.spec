# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for "Current Events".
#
# Build with:   pyinstaller CurrentEvents.spec
#
# This same file works on both macOS and Windows. PyInstaller cannot
# cross-compile, so run it once on a Mac to get the .app and once on a
# Windows machine to get the .exe.

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# ── Source script ───────────────────────────────────────────────────────────
# Change this if your file has a different name/path.
SCRIPT = "CurrentEvents.py"
APP_NAME = "Current Events"

# ── Icon ────────────────────────────────────────────────────────────────────
# Provide icon.icns (macOS) and/or icon.ico (Windows) next to this spec file.
# The correct one is picked automatically for whichever OS you build on.
# Leave the file(s) out and ICON stays None -> the default Python rocket icon.
import os
if sys.platform == "darwin":
    ICON = "icon.icns" if os.path.exists("icon.icns") else None
elif sys.platform == "win32":
    ICON = "icon.ico" if os.path.exists("icon.ico") else None
else:
    ICON = None

# ── Hidden imports & data files ─────────────────────────────────────────────
# pyabf, scipy and matplotlib pull in modules dynamically that PyInstaller's
# static analysis can miss. Collect them explicitly so nothing is left out.
hiddenimports = []
hiddenimports += collect_submodules("pyabf")
hiddenimports += collect_submodules("scipy")
# scipy.signal / scipy.stats sometimes need these explicitly on older builds:
hiddenimports += [
    "scipy._lib.array_api_compat.numpy.fft",
    "scipy.special.cython_special",
]

datas = []
datas += collect_data_files("pyabf")        # pyabf bundles some support files
datas += collect_data_files("matplotlib")   # mpl-data (fonts, styles)

block_cipher = None


a = Analysis(
    [SCRIPT],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim a few large, unused backends to keep the bundle smaller.
    excludes=["tkinter", "PyQt6", "PySide6", "PySide2", "PyQt5.QtWebEngineCore"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX often flags antivirus on Windows; leave off
    console=False,             # GUI app — no terminal window
    disable_windowed_traceback=False,
    argv_emulation=True,       # lets macOS "open with" / file drops work
    target_arch=None,          # set to "universal2" for a fat Mac binary (see notes)
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,                 # set via the ICON variable above
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# On macOS, also wrap the COLLECT output in a proper .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,                 # uses icon.icns when present
        bundle_identifier="com.yourlab.currentevents",
        info_plist={
            "NSHighResolutionCapable": True,   # crisp on Retina displays
            "NSPrincipalClass": "NSApplication",
        },
    )
