import os

pwd = os.environ['PWD']
question_path = os.path.join(pwd, 'dataset/test.json')

print('recommend')
os.system('python3 %s' % (os.path.join(pwd, 'predictor/Ensemble.py' + ' --gpu=0 --question_path=%s' % question_path)))
print('finish')
