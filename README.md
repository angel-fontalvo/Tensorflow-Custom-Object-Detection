Personal project -- Learning how to generate custom object detector models using Tensorflow. For this exercise I used my dog, Harley, as the primary subject to train the model on. 

## Tensorflow Training Process:
1. [Clone the Tensorflow API](#clone-the-tensorflow-api)
2. [Dependencies Installation](#dependency-installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Model Testing](#testing-custom-object-detection)

## Clone the Tensorflow API
1. Clone the [Tensorflow API](https://github.com/tensorflow/models) to your machine.

## Dependency-Installation
1. Follow [these](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) instructions to install all dependencies on your machine.

## Data Preparation:
1.	Manually label the images that contain the object(s) you wish to train. Use the following program to help you with the labelling process: https://github.com/tzutalin/labelImg. The LabelImg app will allow you to label an object within an image, and spit out metadata about that object in an .xml file. 

    **NOTE**: You will need 100 to 500 images to train the model. The more, the better. 

2.	Separate the labeled images into 2 sets: training (90% of total data) & testing (10% of total data).

3.	Create TF records for the datasets:
    
    a.	Convert image .xml data into .csv by running the following script from the terminal:
        
        python xml_to_csv.py
        
    Two files will be generated under the **data** directory named **train_label.csv** and **test_label.csv**.
    
    b.	Once you have converted the files from .xml to .csv, then from the tensorflow/models/research directory, run the following command to install the **object_detection** api:
    
        python setup.py install    
    
    c.  Once the object_detection API has  been installed, then we're ready to generate the TF records. We'll be using the script named **generate_tfrecord.py**. Note:There are a few modifications that must be made: Modify the **row_label** value to be  equals to whatever label you gave the images on step 1. Once you are ready, run the following:
        
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record --data_path=images/test
        
        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --data_path=images/train
        
    A file named **test.record** and **train.record** should have been generated under the **data** directory. These files will be used to train your TensorFlow model. 
    
 ## Model Training
 1. Decide on whether you will use a pre-train model, or training your own from scratch. The benefits of using a pre-trained model, and then use transfer learning to learn a new object is that with transfer learning the training can be much quicker, and the required data need it is much less. You may obtain some pre-train models [through this site](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
 
 2. Decide if you will use [your own custom configuration file](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md), or one that has already been [pre-configure for you](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). Example: If you want to create a model to detect objects real time, use mobilenet. If  you want to simply classify images and speed is not nessessary, but want your model to be a lot more accurate then you may want to use rcnn.
 
 **NOTE**: For this exersice, we're using a pre-configure file (ssd_mobilenet_v1_pets.config) and  it has been placed  under the **training** directory. 
 
 3. Description of some of the values included in the config files:
    
    a. num_classes = Number of objects  you are  training  on, in this example we are only training one object (a chihuahua). 
    
    b. batch_size = how many samples at a time are we throwing at the model to learn against during the training process. If we are getting an out of memory error, try reducing this value. 
    
    c. fine_tune_checkpoint = path to the checkpoint file. 
    
  4. Modify the file named **object-detection.pbtxt** under the **data** directory. This  file contains a dictionary of all objects we  will be training. 
  
     Example:
  
    item {
        id: 1
        name: 'Harley'
    }
    item {
        id: 2
        name: 'Object 2'
    } 
  
  5. Move the following directories and files to the TensorFlow object-detection API ([models/research/object-detection](https://github.com/tensorflow/models/tree/master/research/object_detection)): **data**, **images**, **the directory where you're keeping your model's data (i.e the model's checkponint, frozen_inference_graph.pb, etc)**, and **training**. 
  
  6. If you have not already, add **[{LOCALPATH}/models/research/slim](https://github.com/tensorflow/models/tree/master/research/slim)** to your environment variables as part of **PYTHONPATH**.
  
  7. Once you are ready to train, run the following command (from **{LOCALPATH}/models/research/object_detection**):
  
  ```  
  python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config    
  ```
  
  **NOTE**: As of July 13, 2018, train was moved to the legacy directory. Research how Google recommends you train new models. 
  
  8. If everything worked correctly, you should start seeing the following message repeating (with only the step and loss updating):
  
  ![TensorFlow training your custom object model](/TrainingSample.png?raw=true "Sample Training")
  
  Your goal should be to get your loss to be about 1. Always make sure it's below 2. While it runs, one option you have is to load up Tensorboard to see how to training is doing. To load up Tensorboard, from the object-detection API, run the following:
  
  ```
  tensorboard --logdir=training/
  ```
  
  Running the above script should load up tensorboard on your localhost. And what you want to look out for is mainly **total_loss**. 
  
  ## Model Testing
  
  
