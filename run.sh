#!/bin/bash
cd few-shot_sampling
# python3 sample_few-shot_support_set.py
python3 sample_audio_query_scores.py
python3 sample_image_positives_and_negatives.py
python3 sample_lookups.py
python3 training_and_validation_sample_accuracy.py
python3 image_sample_accuracy.py
cd ../preprocessing/
python3 preprocess_spokencoco_dataset.py
cd ../
python3 run.py