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

If you are running this for a first time you need to provide `unpickle=False` at `sound.py#L15`
By doing so the recognizer class will load all training images from `./train_data` folder and 
will compute vectors for them. Images and labels will be saved to `./saved_data` folder. You can
delete these images now as you don't need them anymore

Later you don't need to load all images and vectors again so you need to provide `unpickle=True` next
time. All images and labels data will be loaded from `./saved_data` folder

Don't forget to change `ANONYMOUS_UPN` and `REST_API_URL` in `settings.py` according to your needs.

### Recognition logic:

Labels data is stored in `dict` format. This is how it is going to look like after first run:

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



