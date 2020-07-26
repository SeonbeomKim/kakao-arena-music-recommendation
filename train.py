import os

pwd = os.environ['PWD']

print('train songs tags artists model')
os.system('python3 %s' % os.path.join(pwd, 'trainer/songs_tags_artists_model_trainer.py' + ' --gpu=0'))

print('train plylst title model')
os.system('python3 %s' % os.path.join(pwd, 'trainer/plylst_title_model_trainer.py' + ' --gpu=0'))

print('finish')
