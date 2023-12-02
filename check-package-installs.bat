@echo off
set packages=llama-cpp-python huggingface_hub llama-index pypdf python-dotenv sentence_transformers datasets loralib sentencepiece einops accelerate langchain bitsandbytes 

for %%p in (%packages%) do (
    conda list | findstr /B /C:"%%p " >nul && (
        echo %%p is installed
    ) || (
        echo %%p is NOT installed
		pip install %%p
    )
) pause