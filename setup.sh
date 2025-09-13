cd ..
git clone https://github.com/565353780/octree-shape.git
git clone https://github.com/565353780/data-convert.git
git clone https://github.com/565353780/base-trainer.git

cd octree-shape
./setup.sh

cd ../data-convert
./setup.sh

cd ../base-trainer
./setup.sh

pip install flash-attn --no-build-isolation

pip install git+https://github.com/mit-han-lab/torchsparse.git
