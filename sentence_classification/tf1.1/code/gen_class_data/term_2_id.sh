input_dir=./test_data/0617_test_tag.txt.next.utf.next.seg
input_dir2=./train-0617-30
len=30
output=./test_data/0617tag-3-30.test
python 3.map_sentence_2_id.py $input_dir $input_dir2"/topic.index" $output $len
