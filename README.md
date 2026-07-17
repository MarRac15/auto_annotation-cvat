This project presents the whole pipeline for training and deploying your custom detection model to CVAT, to enable auto-annotation on your datasets. The scripts were made for detecting the Onion and Weed classes (or Spinach and Weed) specifically, but can be easily tweaked for different classes. For now, it should work with onion and spinach datasets.

### This project consists of the files below:

1. **yolo_label_import.py** - script for converting dataset (labels in XML format) downloaded from CVAT to YOLO format
2. **yolo_model_train** - script for training the chosen yolo model on the given dataset
3. **yolo_eval_test** - for running validation on either val or test dataset

4. **yolo_detect_cvat** - used in the auto-annotation via cvat-cli (more on that later)
5. **yolo_nuctl_function** - handler that uses the trained model to be deployed as a nuclio function (more on that later)
6. **function.yaml** - specification for nuclio associated with the handler above

----

For futher details and instruction on how to use the scripts, please check out the descriptions below:


## 1. Importing labels and prepraring dataset for training:

Download the onion dataset from CVAT (images + labels). To do that:

- Go to task/job and choose *Export dataset* (under 3 dots)
- Select "Save images" and provide a name for your dataset

Save the downloaded zip, preferably in the *cvat_dataset* directory from this project, but it should work with any directory.

To run the script, run in the terminal:

```
python yolo_label_import.py --zip_path <path-to-your-zip> --dataset <crop-type> --out <my_dataset>
```
where:
- < path-to-your-zip > is the path to the directory where you saved your dataset (zip file), e.g.: cvat_dataset/archive.zip
- < my_dataset > is the path for the output directory
- < crop-type > is either *onion* or *spinach* - choose the appropiate option depending on your current dataset type

What the script does is:
- extracts the zip to the folder next to the zip (with the same name)
- makes a split, which results in the creation of train, val and test datasets (0.8/0.1/0.1 on default)
- parses the original XML file and extracts the essential information: only the Onion and Weed classes!
- converts coordinates to YOLO normalised format
- shuffles and splits the data
- copies images and writes label files under:
  ``<out>/images/{train,val,test}/`` and ``<out>/labels/{train,val,test}/``
- Writes a ``data.yaml`` file suitable for YOLO training.

-------

## 2. Training the custom YOLO model

This step requires the dataset in the yolo format from the previous step.

To run the script, run the following command in the terminal:

```
python yolo_model.py --data-path <path-to-dataset-folder> --model <chosen-yolo-model> --epochs <number-of-epochs>
```

where:
- < path-to-dataset-folder > is the path to the output directory of yolo_label.py containing data.yaml
- < chosen-yolo-model > is the variant of the yolo model from ultralytics (yolov8n on default)
- < number-of-epochs > is the number of training epochs

Optional arguments, for more custom training:

- ``--imgsz`` < size-in-pixels > - input image size (pixels), it's 640 on default
- ``--batch`` < batch-size > - batch size used during traing, it's 16 on default
- ``--project`` < path-to-the-results-dir > - project directory for saving results, default path is: *"runs/detect"*
- ``--export`` < output-format > - additional model export format, besides the .pt format (onnx, torchscript, coreml, engine, etc.), default: onnx
- ``--validate-test``- to run validation on test set after training
- ``--no-val`` - to disable validation during training
- ``--no-plots`` - to disable saving plots during training

What the script does:
- it trains the chosen yolo model (yolov8n on default!) on the prepared dataset
- can validate the model on validation or test set
- saves the best model in the chosen output directory (in .pt and .onnx format)

-----

## 3. Running evaluation


----


## 4. Deploying your custom model to CVAT

The free "Community edition" of CVAT enables two ways of using your custom model:

**A) Annotating single tasks via CVAT-CLI**

**B) Using the model straight from the UI**

The first approach is easy and straight-forward, but requires using the terminal which may not be ideal for no technical end-users (such as experts that annotate your data).

The second one is much more elegant, because the model can be used from the UI, although the process of deploying the model is complex and may result in many failed attempts due to the lack of proper instructions in the original documentation. 

----
### The first approach - using CVAT-CLI

Let's start with the first approach, as it is much easier to replicate.


**1) The first step is to write code that will be called by CVAT during auto-annotation.** Not only does it have to run predictions on your model, but also convert the results to the CVAT rectangle annotation format. The documentation specifies that you should create a module, but a file with a single "detect" function would also be enough.

You can find the code in the *yolo_detect_cvat.py* file provided with this project. It's adjusted for our model and the dataset, so it's ready to use!

**2) From now on, you'll be working on the Virtual Machine that runs your CVAT server.** (Skip this step if you're hosting it on your local machine.)

Assuming that you've been training the model on your local machine with GPU, you'll have to copy it to the VM. 

- Copy the YOLO model to the new directory on the VM that runs CVAT:
```
scp <your-local-path-to-the-model> user@vm-ip:/home/user/models/best.pt
```
(You'll have to create the directory first on the VM)

- set the model path to the environmental variable:
```
export CVAT_MODEL_PATH="/home/user/models/best.pt"
```

- Clone this repo:
```
git clone https://github.com/MarRac15/auto_annotation-cvat.git
```

- Create a venv in the project directory and activate it:
```
python -m venv .venv
source .venv/bin/activate
```

- In the venv, install these packages:
``ultralytics``, ``cvat_sdk``


**3) Once you have configured your environment, install the CVAT-CLI:**
```
pip install cvat-cli
```

**4) Authenticate with the CVAT server either via token or password.** It's recommended to use a token, as you don't have to provide your username and password with each command. 

To get the token from the UI, you need to log in to your account, click on your user, choose "profile" and then choose "security". Add a new token on the Personal Access Tokens screen. Be sure to copy the token value.

Export the token as an environmental variable:
```
export CVAT_ACCESS_TOKEN="your_token_value"
```
Cvat reads your token and authenticates you automatically, when it's properly assigned to the environmental variable.


**5)** Create a new project in the CVAT UI, add labels *"Onion"* and *"Weed"*. Create a new task and upload your images to it.


**6) Running the auto-annotation:**

First, run:
```
cvat-cli --server-host <server-address> --server-port <server-port> task ls
```
to find your tasks. Copy the id of the task that you created earlier.


Then, run the auto-annotation on the chosen task:
```
cvat-cli --server-host <server-address> --server-port <server-port> task auto-annotate <your-task-id> --function-file "yolo_detect_cvat.py"
```

You should see the progress bar in the terminal and the appropiate message. Go to the UI to see the results.


**IMPORTANT NOTES!**

- Remember to be in the project directory while executing these commands. Be sure that you are still inside the virtual environment.


- If you encounter *import cv2 error* then:
```
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```
It should fix the problem.


---
**The original documentation:**
- https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/
- https://docs.cvat.ai/docs/api_sdk/cli/

----

### The second approach - via Nuclio

To use your model from the UI, you'll have to deploy it as a Nuclio function (works like a serverless function).

1) Create a new directory on the VM that is running CVAT (e.g. /nuclio-functions). 
Copy the *function.yaml* and *yolo_nuctl_function.py* files from this project to the newly created dir.

**!!! Be sure to go to *function.yaml* file and check the path to the model (in the volumes -> volume -> hostpath -> path)**
Change it to the path where you have your yolo model stored on the VM (e.g.: /home/user/models/best.pt)

2) Stop all the running cvat containers:
```
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down
```

3) Bring up cvat with auto annotation tool. From cvat root directory, run:

```
docker compose -f docker-compose.yml -f components/serverless docker-compose.serverless.yml up -d
```

Run ``docker ps``.
You can see that the nuclio container has just been started alongside the cvat container


4) Install nuctl - a command line tool for deploying nuclio functions:

```
wget https://github.com/nuclio/nuclio/releases/download/1.16.3/nuctl-1.16.3-linux-amd64
```

It is important that the version you download matches the version in docker-compose.serverless.yml

In my case it’s version 1.16.3, if yours is different then change it in the github link in the command above.

5) After downloading the nuclio, give it a proper permission and do a softlink:

```
sudo chmod +x nuctl-1.16.3-linux-amd64

sudo ln -sf $(pwd)/nuctl-1.16.3-linux-amd64 /usr/local/bin/nuctl
```

(Again - change the version if necessary)


6) Go to the */nuclio-functions* that you created in the first step. 


7) Create a new nuclio project:
```
nuctl create project cvat --platform local
```

8) Deploy the function:
```
nuctl deploy --project-name cvat --path . --platform local --verbose
```

It detects your *.yaml* file and *.py* handler automatically (if you're in this directory that is).
The deployment process can take a while.


9) Check the status of your function:
```
nuctl get functions
```

You should see it listed below with state *"ready"*.

Go to the UI, go to the *Models* page and you'll see your model.

From now on you'll be able to use it straight from the UI!
Go to your tasks, click on *"Actions"* next to your task and select *"Autoamtic annotation"* option.

---

Official documentations:
- https://docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/
- https://docs.cvat.ai/docs/guides/serverless-tutorial/
