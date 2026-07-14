#!/usr/bin/env bash

MODE=$1
INPUT=$2
OUTPUT_DIR=$3

if [ "$MODE" == "data/train_data/" ] || [ -d "$MODE" ]; then
    echo "Training Word2Vec model..."
    TRAIN_DATA_PATH=$MODE
    python3 train_word2vec.py --data_path "$TRAIN_DATA_PATH"
    echo "Training completed. Model saved to models/"

elif [ "$MODE" == "test1" ]; then
    echo "Running Task 1 predictions..."
    TEST_FILE=$INPUT
    PREDICTIONS_DIR=$OUTPUT_DIR

    mkdir -p "$PREDICTIONS_DIR"
    python3 task1.py --input "$TEST_FILE" --output "$PREDICTIONS_DIR/task1_predictions.jsonl"
    echo "Task 1 predictions saved to $PREDICTIONS_DIR/task1_predictions.jsonl"

elif [ "$MODE" == "test2" ]; then
    echo "Running Task 2 predictions..."
    TEST_FILE=$INPUT
    PREDICTIONS_DIR=$OUTPUT_DIR

    mkdir -p "$PREDICTIONS_DIR"
    python3 task2.py --input "$TEST_FILE" --output "$PREDICTIONS_DIR/task2_predictions.json"
    echo "Task 2 predictions saved to $PREDICTIONS_DIR/task2_predictions.json"

else
    echo "Invalid mode. Usage:"
    echo "  bash run_model.sh data/train_data/"
    echo "  bash run_model.sh test1 task1_test.json predictions/"
    echo "  bash run_model.sh test2 task2_test.json predictions/"
    exit 1
fi
