@echo off
echo ===================================================
echo   Starting Full AI Model Training
echo ===================================================
echo.
echo NOTE: This will train the CNN model on all 54,000 images!
echo It may take several hours depending on your computer's speed.
echo Please leave this window open until it says "Training Complete."
echo.

python training/train_model.py

echo.
echo ===================================================
echo Training has finished! The new, highly accurate model
echo has been saved to the 'models' folder.
echo You can now use run.bat to start the website and test it.
echo ===================================================
pause
