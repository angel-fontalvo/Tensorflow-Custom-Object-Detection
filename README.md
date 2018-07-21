Tensorflow Training Process (from scratch):
1.	Manually label the images. 
2.	Separate the label images into 2 sets: training (90% of total data) & testing (10% of total data).
3.	Create TF records for the datasets:
    
    a.	Convert image .xml data into .csv by running the following script:
        
        xml_to_csv.py
    
    b.	Once you have converted the files from .xml to .csv, then create the TFRecord from the .csv file. Use the python script named generate_tfrecord.py. There are a few modifications that must be made:
    
        i.	Modify the ##row_label## value to be  equals to whatever label you gave the images on step 1. 

    c. Once you are ready to generate the TF record, run the following:
        
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
        
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
        
    A file named ##test.record## and ##train.record## should have been generated under the data directory. These files will be used to train your TensorFlow model. 
