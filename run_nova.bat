@echo off
REM Launch Nova with the real Python, working around the Windows Store "python" alias.
REM If you later disable the alias (Settings > Apps > Advanced app settings >
REM App execution aliases > turn off python.exe), plain "python nova.py" works too.
set "NOVA_PY=C:\Users\ritas\AppData\Local\Python\bin\python.exe"
if not exist "%NOVA_PY%" set "NOVA_PY=python"
"%NOVA_PY%" "%~dp0nova.py" %*
