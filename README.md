## Tensorflow Training Process (from scratch):
1. [Dependencies Installation](#dependency-installation)
2. [Data Gathering](#data-gathering)
3. [Model Training](#model-training)

## Dependency-Installation
1. Follow [these](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) instructions.

## Data Gathering:
1.	Manually label the images that contain the object(s) you wish to train. Use the following program to help you with the labelling process: https://github.com/tzutalin/labelImg. The LabelImg app will allow you to label an object within an image, and spit out metadata about that object in a .xml file. 
2.	Separate the labeled images into 2 sets: training (90% of total data) & testing (10% of total data).
3.	Create TF records for the datasets:
    
    a.	Convert image .xml data into .csv by running the following script:
        
        python xml_to_csv.py
    
    b.	Once you have converted the files from .xml to .csv, then from the tensorflow/models/research directory, run the following command to install the **object_detection** api:
    
        python setup.py install    
    
    c.  Once the object_detection API has  been installed, then we're ready to generate the TF records. We'll be using the script named **generate_tfrecord.py**. Note:There are a few modifications that must be made: Modify the **row_label** value to be  equals to whatever label you gave the images on step 1. Once you are ready, run the following:
        
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
        
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
        
    A file named **test.record** and **train.record** should have been generated under the data directory. These files will be used to train your TensorFlow model. 
    
 ## Model Training
