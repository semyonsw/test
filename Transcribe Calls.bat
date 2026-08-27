@echo off
rem Double-click me. Optionally drag a folder of recordings onto me to browse it.
title Armenian Call Transcriber
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_stt.ps1" %*
