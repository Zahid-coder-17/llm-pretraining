@echo off
setlocal
echo ===================================================
echo     HyperGQE Automated Data Pipeline (Offline)
echo ===================================================
echo.

echo [1/8] Cleaning Data...
python clean_data.py
if %ERRORLEVEL% NEQ 0 goto :error
echo ^> Deleting raw data to save space...
rmdir /s /q "E:\pretrain_data\raw"

echo.
echo [2/8] Deduplicating Data (This takes time/RAM)...
python deduplicate.py
if %ERRORLEVEL% NEQ 0 goto :error
echo ^> Deleting processed data to save space...
rmdir /s /q "E:\pretrain_data\processed"

echo.
echo [3/8] Splitting Data into Train/Val/Test...
python split_data.py
if %ERRORLEVEL% NEQ 0 goto :error
echo ^> Deleting deduped data to save space...
rmdir /s /q "E:\pretrain_data\deduped"

echo.
echo [4/8] Training BPE Tokenizer...
python train_tokenizer.py
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [5/8] Tokenizing Data...
python tokenize_data.py
if %ERRORLEVEL% NEQ 0 goto :error
echo ^> Deleting text splits to save space...
rmdir /s /q "E:\pretrain_data\splits"

echo.
echo [6/8] Packing Sequences (512 length)...
python pack_sequences.py
if %ERRORLEVEL% NEQ 0 goto :error
echo ^> Deleting raw tokenized data to save space...
rmdir /s /q "E:\pretrain_data\tokenized"

echo.
echo [7/8] Building Manifests...
python build_manifest.py
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo [8/8] Validating Final Dataset...
python validate_dataset.py
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo ===================================================
echo     PIPELINE COMPLETED SUCCESSFULLY!
echo ===================================================
echo.
echo E: Drive is now fully cleaned up and contains only
echo the final packed shards in E:\pretrain_data\train\ (and val/test)
echo.
echo You can now start training:
echo   python train.py --arch gqa --seed 42
echo   python train.py --arch gqe --seed 42
echo   python train.py --arch hyper_gqe --seed 42
goto :EOF

:error
echo.
echo [ERROR] Pipeline failed at the current step.
echo Check the logs in the terminal above.
exit /b 1
