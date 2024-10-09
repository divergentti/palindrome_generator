# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PalindromiPeli.py'],
    pathex=[],
    binaries=[],
    datas=[
 ('data/adjektiivi_sanat.csv', 'data'), 
 ('data/verbi_sanat.csv', 'data'),
 ('data/adj_palindromes.csv', 'data'), 
 ('data/subs_palindromes.csv','data'), 
 ('data/verb_palindromes.csv','data'),
 ('data/ext_palindromes.csv', 'data'), 
 ('data/substantiivi_sanat.csv', 'data'),
 ('data/palindromes.json', 'data'), 
 ('data/runtimeconfig.json', 'data'),
 ('data/LongText.txt', 'data'),
 ('README.md', '.') 
 ],
    hiddenimports=[
        'PyQt6',
        'matplotlib',
        'nltk',
        'pandas',
        'scikit-learn',
        'qasync',
        'gensim'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PalindromiPeli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
