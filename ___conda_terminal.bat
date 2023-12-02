@echo off
CALL C:\ProgramData\anaconda3\Scripts\activate.bat
CALL activate open-source-llm
cd %~dp0
Rem keep window open for prompting
cmd /k
