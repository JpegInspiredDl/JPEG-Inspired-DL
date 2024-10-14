# fetch pre-trained teacher models

mkdir -p save/models/

cd save/models

mkdir -p resnet56_vanilla
wget http://shape2prog.csail.mit.edu/repo/resnet56_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth resnet56_vanilla/

mkdir -p resnet110_vanilla
wget http://shape2prog.csail.mit.edu/repo/resnet110_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth resnet110_vanilla/

mkdir -p vgg13_vanilla
wget http://shape2prog.csail.mit.edu/repo/vgg13_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth vgg13_vanilla/


cd ../..