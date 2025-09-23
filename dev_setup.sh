cd ..
git clone git@github.com:565353780/sdf-generate.git
git clone git@github.com:565353780/data-convert.git
git clone git@github.com:565353780/base-trainer.git
git clone git@github.com:565353780/point-cept.git

cd sdf-generate
./dev_setup.sh

cd ../data-convert
./dev_setup.sh

cd ../base-trainer
./dev_setup.sh

cd ../point-cept
./dev_setup.sh

pip install flash-attn --no-build-isolation

pip install git+https://github.com/mit-han-lab/torchsparse.git
