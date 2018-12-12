
python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

mv trained_model training_1

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/2"

mv trained_model training_2

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

mv trained_model training_3

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/2"

mv trained_model training_4

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/2"

mv trained_model training_5

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.0001         --how_many_training_steps=60000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224  --print_misclassified_test_images    --tfhub_module="https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2"

mv trained_model training_6
