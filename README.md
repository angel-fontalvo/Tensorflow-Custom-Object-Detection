Tensorflow Training Process (from scratch):
1.	Manually label the images. 
2.	Separate the label images into 2 sets: training (90% of total data) & testing (10% of total data).
3.	Create TF records for the datasets:
    
    a.	Convert image .xml data into .csv (use git repo: https://github.com/datitran/raccoon_dataset to do so). Use the file  named xml_to_csv.py to parse through each xml file and convert it to csv. There are a few modifications that must be made, refer to this video to see what they are: https://www.youtube.com/watch?v=kq2Gjv_pPe8.  
    
    b.	Once you have converted the files from .xml to .csv, then create the TFRecord from the .csv file. Use the file named generate_tfrecord.py from the above repo. There are a few modifications that must be made:
    
        i.	Modify the ##row_label## value to be  equals to whatever label you gave the images on step 1. 

    c. Once you are ready to generate the TF record, run the following:
        
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
        
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
        
    A file named ##test.record## and ##train.record## should have been generated under the data directory. These files will be used to train your TensorFlow model. 
