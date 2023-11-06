@echo off
call %userprofile%\Anaconda3\condabin\activate.bat
call %userprofile%\Anaconda3\condabin\conda.bat activate open-source-llm
d:
cd Git
cd open-source-llm
python mistral-rag-3.py
pause