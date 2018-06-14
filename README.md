# Quest Coffee Machine

![alt text](https://i.imgur.com/kVR707K.jpg)

### Requirements:

- Raspberry Pi 3 Model B
- One sound sensor connected to GPIO pins
- One coffee machine 

### How to run:

Assuming you have clean RPi system image installed

Clone this repo first:

```
git clone https://github.com/wkpn/quest-coffee-machine
```

First thing you need to do is to place your training images inside `./train_data` folder. Each 
image should be in `label`.jpg format. One image per person

```
train_data
├── Label1.jpg
├── Label2.jpg
├── Label3.jpg
├── ...
```

Next you need to run some scripts

```sh
$ cd ./libs
$ ./download_libs.sh
```

This script will download all the necessary libraries for the whole thing to function. When 
it's done you need to run script which is going to download `face recognition` model along 
with `shape predictor` model

```sh
$ cd ./models
$ ./download_models.sh
```

At the very first run you can execute `setup.py` script. This script will go through all your images in `./train_data` 
folder and will compute vectors for them. It will save all data about `labels` in `./labels` folder. 
If you do this, you need to provide `unpickle=True` at `main.py#L14`. It means that the recognizer will load data about 
labels from `setup.py` results. You can delete your `./train_data` folder after running this script

It is not necessary to run `setup.py`. If you won't run it before running `main.py` you need to provide `unpickle=False` 
at `main.py#L14`. By doing so the recognizer class will load all training images from `./train_data` folder and 
will compute vectors for them. Labels will be saved to `./labels` folder. You can delete these images now 
as you don't need them anymore. Later you need to provide `unpickle=True` at `main.py#L14`. All labels data will be 
loaded from `./saved_data` folder

Don't forget to change `ANONYMOUS_UPN` and `REST_API_URL` in `settings.py` according to your needs!

### Recognition logic:

Labels data is stored in `dict` format. This is how it is going to look like after first run (or running `setup.py`):

```python
{
  'Label1': [Label1Vector1],
  'Label2': [Label2Vector1],
  ...
}

```

Model is learning after each successful face recognition. For example, if we successfully recognized 
a person we are going to add his vector to `labels` dictionary. After some time it will be looking 
like that:

```python
{
  'Label1': [Label1Vector1, Label1Vector2, Label1Vector3],
  'Label2': [Label2Vector1], Label2Vector2 ...]
  ...
}
```

It means that model is learning and each time recognition will be more accurate. We save our labels 
after each successful face recognition for two reasons:

- We can stop the whole thing and run it next time with new data
- Backup

That's it. You're ready to go!



