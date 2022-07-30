# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['Main_UI.py','ColorMenu.py','edgeMenu.py','faceDetect.py','filterMenu.py','geometryMenu.py','mophologyMenu.py','ui_file_1.py'],
    pathex=['C:\\Users\\Simon\\Desktop\\DigitalImageProject\\DILProject'],
    binaries=[],
    datas=[('C:\\Users\\Simon\\Desktop\\DigitalImageProject\\DILProject\\OriginPictures','OriginPictures'),('C:\\Users\\Simon\\Desktop\\DigitalImageProject\\DILProject\\SavedPictures','SavedPictures'),('C:\\Users\\Simon\\Desktop\\DigitalImageProject\\DILProject\\img','img')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='Main_UI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Main_UI',
)
