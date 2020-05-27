export GLUE_DIR=../datas
export TASK_NAME=bwccsents

python ../examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name garbledsents \
    --do_train True \
    --do_eval True \
    --do_test False \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_train_batch_size=32   \
    --per_gpu_eval_batch_size=64   \
    --per_gpu_test_batch_size=64   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --n_gpu 2 \
    --output_dir ../output/$TASK_NAME/