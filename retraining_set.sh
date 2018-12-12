
python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"

mv trained_model training_1

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"

mv trained_model training_2

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

mv trained_model training_3

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

mv trained_model training_4

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1"

mv trained_model training_5

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=299 --input_width=299      --tfhub_module="https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1"

mv trained_model training_6

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=331 --input_width=331      --tfhub_module="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1"

mv trained_model training_7

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=331 --input_width=331      --tfhub_module="https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1"

mv trained_model training_8

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=331 --input_width=331      --tfhub_module="https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2"

mv trained_model training_9

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=331 --input_width=331      --tfhub_module="https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2"

mv trained_model training_10

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=16        --bottleneck_dir="data/bottleneck"         --learning_rate=0.01         --how_many_training_steps=2000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224      --tfhub_module="https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1"

mv trained_model training_11

python3 retrain.py         --image_dir="data/category"         --output_graph="trained_model/final_model.pb"              --output_labels="trained_model/final_label.pbtxt"         --summaries_dir="trained_model/retrain_logs"         --train_batch_size=8        --bottleneck_dir="data/bottleneck"         --learning_rate=0.001         --how_many_training_steps=10000  --validation_percentage=20 --testing_percentage=20 --input_height=224 --input_width=224      --tfhub_module="https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1"

mv trained_model training_12
