REM #train_detector model trains the model with the output images of classification.py
REM #			and generates detection models and tensor flow plots


python  train_detector.py --d  "output\dataset_mask" --p "output\mask_plot.png" --m "output\maskdetector.model"

python  train_detector.py --d  "output\dataset_helmet" --p "output\helmet_plot.png" --m "output\helmetdetector.model"

    