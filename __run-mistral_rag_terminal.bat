@echo off
CALL C:\ProgramData\anaconda3\Scripts\activate.bat
CALL activate open-source-llm
cd %~dp0
python _mistral-rag-document-summary-llama_cpp.py
pause



